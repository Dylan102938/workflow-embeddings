"""Frozen text embeddings from a local model: instruction keys, paraphrase keys and typed-text vectors.

  WFEMB_KEYS=<key set dir> python -m wfemb.prep.text_embed [--model Qwen/Qwen3-Embedding-0.6B] [--para_keep_frac F]

File formats: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable

import numpy as np
import torch

from ..storage import paths
from . import key_set
from .paraphrase import PARAPHRASES_PATH, load_paraphrases
from .prepare import load_sessions

INSTRUCTION_MAX_TOKENS = 256
TYPED_TEXT_MAX_TOKENS = 128

_LITERAL_PATTERNS = [
    (
        re.compile(r"\b[\w\-]+\.(?:xlsx?|docx?|pptx?|pdf|csv|txt|png|jpe?g|py|json|md|zip|mp4|mp3|html?)\b", re.I),
        "<FILE>",
    ),
    (re.compile(r"\"[^\"]{1,80}\"|'[^']{1,80}'"), "<VALUE>"),
    (re.compile(r"\b\d+(?:[.,:/-]\d+)*\b"), "<VALUE>"),
]


def main() -> None:
    args = parse_arguments()
    refuse_to_overwrite_default_key_set(args.overwrite_default)
    print(f"writing key set to {paths.KEYS} (model {args.model})", flush=True)
    paths.KEYS.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions()
    typed_texts = load_typed_texts()
    rewrite: Callable[[str], str] = replace_literals if args.placeholders else (lambda text: text)
    model = load_embedder(args.model, INSTRUCTION_MAX_TOKENS)

    keys = embed_instruction_keys(model, [rewrite(session["instruction"]) for session in sessions], args.batch)
    stats = {
        "model": args.model,
        "dim": int(keys.shape[1]),
        "placeholders": args.placeholders,
        "theta_para": args.theta_para,
    }

    if PARAPHRASES_PATH.exists():
        stats.update(embed_paraphrase_keys(model, sessions, keys, rewrite, args))
        print("paraphrases", stats)
    else:
        print("no paraphrases.json yet; skipping paraphrase keys")

    if not args.skip_texts:
        embed_typed_texts(model, typed_texts, keys.shape[1], args.batch)
        stats["n_texts"] = len(typed_texts)
    with open(paths.KEYS / "meta.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--theta_para", type=float, default=0.85)
    parser.add_argument(
        "--placeholders",
        action="store_true",
        help="embed instructions with incidental literals replaced by placeholders",
    )
    parser.add_argument("--skip_texts", action="store_true")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--overwrite_default",
        action="store_true",
        help="allow writing the default key set (paths.SHARED); otherwise set WFEMB_KEYS",
    )
    parser.add_argument(
        "--para_keep_frac",
        type=float,
        default=0.0,
        help="if set, choose theta_para as the similarity quantile that keeps this fraction "
        "(calibrates the drift filter to a new embedder's similarity scale)",
    )
    return parser.parse_args()


def refuse_to_overwrite_default_key_set(overwrite_allowed: bool) -> None:
    writes_default = paths.KEYS.resolve() == paths.SHARED.resolve() and (paths.SHARED / "keys.npy").exists()
    if writes_default and not overwrite_allowed:
        raise SystemExit(
            f"refusing to overwrite the default key set in {paths.SHARED}; set WFEMB_KEYS to a new "
            f"directory (or pass --overwrite_default)"
        )


def load_typed_texts() -> list[str]:
    with open(paths.PREP / "vocab.json") as f:
        return json.load(f)["texts"]


def replace_literals(text: str) -> str:
    """File names, quoted strings and numbers become <FILE> / <VALUE>."""
    for pattern, placeholder in _LITERAL_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def load_embedder(name: str, max_tokens: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = max_tokens
    return model


def encode(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """L2-normalized float32 embeddings, in the order of texts."""
    longest_first = np.argsort([len(text) for text in texts])[::-1]  # similar lengths share a batch
    sorted_vectors = model.encode(
        [texts[i] for i in longest_first],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.empty_like(sorted_vectors)
    vectors[longest_first] = sorted_vectors
    return vectors.astype(np.float32)


def encode_deduplicated(model, texts: list[str], batch_size: int) -> np.ndarray:
    unique_texts, positions = key_set.deduplicate(texts)
    return encode(model, unique_texts, batch_size)[positions]


def embed_instruction_keys(model, instructions: list[str], batch_size: int) -> np.ndarray:
    """Writes keys.npy (one row per session) and returns it."""
    keys = encode_deduplicated(model, instructions, batch_size)
    np.save(paths.KEYS / "keys.npy", keys)
    print("keys", keys.shape)
    return keys


def embed_paraphrase_keys(
    model, sessions: list[dict], keys: np.ndarray, rewrite: Callable[[str], str], args: argparse.Namespace
) -> dict:
    """Writes para_keys.npy and para_index.npz (paraphrases that pass the drift filter); returns their stats."""
    texts, owners = key_set.session_paraphrases(sessions, load_paraphrases())
    vectors = encode_deduplicated(model, [rewrite(text) for text in texts], args.batch)
    similarities = key_set.drift_similarities(vectors, owners, keys)
    threshold = args.theta_para
    if args.para_keep_frac:
        threshold = key_set.threshold_keeping(similarities, args.para_keep_frac)
    return key_set.write_paraphrase_keys(paths.KEYS, vectors, owners, similarities, threshold, len(sessions))


def embed_typed_texts(model, typed_texts: list[str], dim: int, batch_size: int) -> None:
    """Writes text_vecs.npy: one row per typed string of vocab.json."""
    model.max_seq_length = TYPED_TEXT_MAX_TOKENS
    vectors = encode(model, typed_texts, 2 * batch_size) if typed_texts else np.zeros((0, dim), np.float32)
    np.save(paths.KEYS / "text_vecs.npy", vectors.astype(np.float16))


if __name__ == "__main__":
    main()
