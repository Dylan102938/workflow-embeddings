"""Throughput check for the frame encoder on synthetic screenshots (run under srun on rnn)."""
import time
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, Siglip2VisionModel
from wfemb.cache_frames import pool_grid

name = "google/siglip2-so400m-patch16-naflex"
proc = AutoImageProcessor.from_pretrained(name)
vis = Siglip2VisionModel.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").vision_model.cuda().eval()
imgs = [Image.fromarray(np.random.randint(0, 255, (1080, 1920, 3), np.uint8)) for _ in range(32)]
t = time.time(); g = proc(images=imgs, max_num_patches=1024, return_tensors="pt"); tp = time.time() - t
crops = [im.crop((0, 0, 480, 480)).resize((384, 384)) for im in imgs]
c = proc(images=crops, max_num_patches=576, return_tensors="pt")
print("proc s/img", tp / 32, {k: tuple(v.shape) for k, v in g.items()}, g["spatial_shapes"][0].tolist(), c["spatial_shapes"][0].tolist())
with torch.inference_mode():
    for it in range(4):
        torch.cuda.synchronize(); t = time.time()
        h = vis(pixel_values=g["pixel_values"].cuda().bfloat16(), attention_mask=g["pixel_attention_mask"].cuda(), spatial_shapes=g["spatial_shapes"].cuda()).last_hidden_state
        hc = vis(pixel_values=c["pixel_values"].cuda().bfloat16(), attention_mask=c["pixel_attention_mask"].cuda(), spatial_shapes=c["spatial_shapes"].cuda()).last_hidden_state
        p = pool_grid(h, g["spatial_shapes"], (8, 8)); pc = pool_grid(hc, c["spatial_shapes"], (4, 4))
        torch.cuda.synchronize(); print("gpu frames/s", 32 / (time.time() - t), tuple(p.shape), tuple(pc.shape), h.dtype)
print("mem GB", torch.cuda.max_memory_allocated() / 1e9)
