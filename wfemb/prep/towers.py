"""The frozen vision towers whose native features are cached: screenshots on the GPU -> one token per output cell.

Each tower's preprocessing and what its tokens are: docs/DATA_PREP.md.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.request

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as transforms

DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_BYTES = 64 << 20  # per request, so that a stalled connection costs one piece
INKLING_WEIGHT_NAMES = {  # checkpoint name -> name in transformers' Inkling2VisionModel
    r"^backbone\.embeddings\.position_embedding$": "embeddings.position_embedding.weight",
    r"^backbone\.embeddings\.": "embeddings.",
    r"^backbone\.ln_pre\.": "pre_layernorm.",
    r"^backbone\.layers\.": "layers.",
    r"^proj_in\.": "merger.linear_1.",
    r"^proj_out\.": "merger.linear_2.",
    r"\.ln_1\.": ".layernorm_before.",
    r"\.ln_2\.": ".layernorm_after.",
    r"\.mlp\.c_fc\.": ".mlp.fc1.",
    r"\.mlp\.c_proj\.": ".mlp.fc2.",
    r"\.ls_1\.gamma$": ".lambda_1",
    r"\.ls_2\.gamma$": ".lambda_2",
}


def grid_size(height: int, width: int, stride: int, budget: int) -> tuple[int, int]:
    """Rows and columns of output tokens: the image's own size in strides, scaled down if that exceeds the budget."""
    rows, columns = max(1, round(height / stride)), max(1, round(width / stride))
    if rows * columns > budget:
        scale = math.sqrt(height * width / (budget * stride**2))
        rows, columns = max(1, math.floor(height / scale / stride)), max(1, math.floor(width / scale / stride))
    return rows, columns


class Tower:
    """Subclasses set repository, stride, dim and interpolation, and implement load() and features()."""

    repository: str
    stride: int  # pixels per output token, each way
    dim: int  # channels per output token
    interpolation: InterpolationMode  # the resize filter of the tower's own processor

    def __init__(self, budget: int, device: str = "cuda"):
        self.budget, self.device = budget, device
        self.model = self.load().to(device, torch.bfloat16).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """uint8 screenshots [B, 3, H, W] of one size -> bfloat16 tokens [B, rows * columns, dim], (rows, columns)."""
        rows, columns = grid_size(*images.shape[-2:], self.stride, self.budget)
        size = [rows * self.stride, columns * self.stride]
        resized = transforms.resize(images.to(self.device), size, self.interpolation, antialias=True)
        pixels = (resized.float() / 127.5 - 1).to(self.model.dtype)
        return self.features(pixels, rows, columns).to(torch.bfloat16), (rows, columns)

    def grids(self, batch_size: int, rows: int, columns: int) -> torch.Tensor:
        return torch.tensor([[1, rows, columns]] * batch_size, device=self.device)


def attention_per_image(queries, keys, values, n_images: int) -> torch.Tensor:
    """Self-attention within each of n_images equally long images packed in one sequence.

    [sequence, heads, d] each -> [sequence, heads * d]. transformers runs one attention call per image instead.
    """
    length = queries.shape[0] // n_images
    queries, keys, values = (x.view(n_images, length, *x.shape[1:]).transpose(1, 2) for x in (queries, keys, values))
    return F.scaled_dot_product_attention(queries, keys, values).transpose(1, 2).flatten(0, 1).flatten(1)


def patches(pixels: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, H / patch, W / patch, patch, patch, C]"""
    batch, channels, height, width = pixels.shape
    cut = pixels.reshape(batch, channels, height // patch, patch, width // patch, patch)
    return cut.permute(0, 2, 4, 3, 5, 1)


class Siglip2(Tower):
    """SigLIP2 So400m NaFlex: 16-pixel patches, averaged in 2 x 2 groups."""

    repository, stride, dim, interpolation = (
        "google/siglip2-so400m-patch16-naflex",
        32,
        1152,
        InterpolationMode.BILINEAR,
    )

    def load(self):
        from transformers import Siglip2VisionModel

        return Siglip2VisionModel.from_pretrained(self.repository, dtype=torch.bfloat16)

    def features(self, pixels, rows, columns):
        batch = pixels.shape[0]
        shapes = torch.tensor([[2 * rows, 2 * columns]] * batch, device=pixels.device)
        hidden = self.model(
            pixel_values=patches(pixels, 16).flatten(3).flatten(1, 2),
            pixel_attention_mask=torch.ones(batch, 4 * rows * columns, dtype=torch.int32, device=pixels.device),
            spatial_shapes=shapes,
        ).last_hidden_state
        return hidden.float().view(batch, rows, 2, columns, 2, -1).mean((2, 4)).flatten(1, 2)


class Qwen38(Tower):
    """Qwen3.8-27B's vision tower: 16-pixel patches, its 2 x 2 patch merger and projector."""

    repository, stride, dim, interpolation = "Qwen/Qwen3.8-27B", 32, 5120, InterpolationMode.BICUBIC

    def load(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

        def attention(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
            heads = self.qkv(hidden_states).view(len(hidden_states), 3, self.num_heads, -1).unbind(1)
            queries, keys = qwen.apply_rotary_pos_emb_vision(heads[0], heads[1], *position_embeddings)
            return self.proj(attention_per_image(queries, keys, heads[2], len(cu_seqlens) - 1))

        qwen.Qwen3_5VisionAttention.forward = attention
        model = Qwen3_5VisionModel(Qwen3_5VisionConfig(**checkpoint_config(self.repository)["vision_config"]))
        model.load_state_dict(checkpoint_tensors(self.repository, "model.visual."))
        return model

    def features(self, pixels, rows, columns):
        batch = pixels.shape[0]
        frames = pixels[:, :, None].expand(-1, -1, 2, -1, -1)  # an image is a video of two equal frames
        cut = frames.reshape(batch, 3, 2, rows, 2, 16, columns, 2, 16).permute(0, 3, 6, 4, 7, 1, 2, 5, 8)
        merged = self.model(cut.reshape(-1, 3 * 2 * 16 * 16), self.grids(batch, 2 * rows, 2 * columns)).pooler_output
        return merged.view(batch, rows * columns, -1)


class Inkling2(Tower):
    """Inkling 2 Small's vision tower: 20-pixel patches, its 2 x 2 patch merger and projector."""

    repository, stride, dim, interpolation = "thinkingmachines/Inkling-2-Small", 40, 4608, InterpolationMode.BICUBIC

    def load(self):
        from transformers.models.inkling2 import modeling_inkling2 as inkling
        from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionConfig, Inkling2VisionModel

        def attention(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
            projections = (self.q_proj, self.k_proj, self.v_proj)
            heads = [project(hidden_states).view(len(hidden_states), self.num_heads, -1) for project in projections]
            queries, keys = inkling.apply_rotary_pos_emb_vision(heads[0], heads[1], *position_embeddings)
            return self.out_proj(attention_per_image(queries, keys, heads[2], len(cu_seqlens) - 1)), None

        inkling.Inkling2VisionAttention.forward = attention
        model = Inkling2VisionModel(Inkling2VisionConfig(**checkpoint_config(self.repository)["vision_config"]))
        tensors = checkpoint_tensors(self.repository, "model.visual.")
        for pattern, replacement in INKLING_WEIGHT_NAMES.items():
            tensors = {re.sub(pattern, replacement, name): tensor for name, tensor in tensors.items()}
        model.load_state_dict(tensors)
        return model

    def features(self, pixels, rows, columns):
        batch = pixels.shape[0]
        merged = self.model(patches(pixels, 40).flatten(0, 2), self.grids(batch, rows, columns)).pooler_output
        return merged.view(batch, rows * columns, -1)


TOWERS = {"siglip2": Siglip2, "qwen3.8": Qwen38, "inkling2": Inkling2}


def checkpoint_file(repository: str, file: str, first: int = 0, end: int | None = None) -> bytes:
    """Bytes [first, end) of a file of a public Hugging Face repository."""
    request = urllib.request.Request(
        f"https://huggingface.co/{repository}/resolve/main/{file}",
        headers={"Range": f"bytes={first}-{'' if end is None else end - 1}"},
    )
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:  # a stalled or dropped connection
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(5)


def checkpoint_config(repository: str) -> dict:
    return json.loads(checkpoint_file(repository, "config.json"))


def checkpoint_tensors(repository: str, prefix: str) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors whose names start with prefix, without it.

    Read with ranged requests: a tower is about 1 GB of a checkpoint of 55 GB (Qwen) or 390 GB (Inkling).
    """
    index = json.loads(checkpoint_file(repository, "model.safetensors.index.json"))
    tensors = {}
    for file in sorted({file for name, file in index["weight_map"].items() if name.startswith(prefix)}):
        header_size = int.from_bytes(checkpoint_file(repository, file, 0, 8), "little")
        header = json.loads(checkpoint_file(repository, file, 8, 8 + header_size))
        wanted = {name: entry for name, entry in header.items() if name.startswith(prefix)}
        first = min(entry["data_offsets"][0] for entry in wanted.values())
        last = max(entry["data_offsets"][1] for entry in wanted.values())
        data = bytearray()
        for start in range(8 + header_size + first, 8 + header_size + last, DOWNLOAD_BYTES):
            data += checkpoint_file(repository, file, start, min(start + DOWNLOAD_BYTES, 8 + header_size + last))
        for name, entry in wanted.items():
            if entry["dtype"] != "BF16":
                raise ValueError(f"{repository} {name}: {entry['dtype']} weights are not supported")
            start, end = entry["data_offsets"]
            values = torch.frombuffer(data, dtype=torch.bfloat16, count=(end - start) // 2, offset=start - first)
            tensors[name.removeprefix(prefix)] = values.reshape(entry["shape"])
    return tensors
