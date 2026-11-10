"""Label key set from OpenAI embeddings: instruction keys and paraphrase keys (training / evaluation targets).

  python -m wfemb.prep.openai_keys [--model text-embedding-3-large] [--out DATA/keys_openai] [--para_keep_frac 0.927]

Needs OPENAI_API_KEY. File formats: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..storage import paths
from . import key_set
from .paraphrase import PARAPHRASES_PATH, load_paraphrases
from .prepare import load_sessions

USD_PER_MILLION_TOKENS = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}
DEFAULT_USD_PER_MILLION_TOKENS = 0.13
MAX_INPUT_CHARACTERS = 30000  # keeps every input under the API's 8191-token limit


def main() -> None:
    args = parse_arguments()
    from openai import OpenAI

    client = OpenAI(max_retries=10)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions()
    paraphrases = load_paraphrases()

    instructions = [session["instruction"] for session in sessions]
    keys, instruction_tokens = embed_each(client, args, instructions, output_dir / "raw_instr")
    np.save(output_dir / "keys.npy", keys.astype(np.float32))
    print("keys", keys.shape, flush=True)

    paraphrase_texts, owners = key_set.session_paraphrases(sessions, paraphrases)
    paraphrase_vectors, paraphrase_tokens = embed_each(client, args, paraphrase_texts, output_dir / "raw_para")
    similarities = key_set.drift_similarities(paraphrase_vectors, owners, keys)
    threshold = key_set.threshold_keeping(similarities, args.para_keep_frac)
    paraphrase_stats = key_set.write_paraphrase_keys(
        output_dir, paraphrase_vectors, owners, similarities, threshold, len(sessions)
    )
    link_typed_text_vectors(output_dir)

    api_tokens = instruction_tokens + paraphrase_tokens
    usd_per_million_tokens = USD_PER_MILLION_TOKENS.get(args.model, DEFAULT_USD_PER_MILLION_TOKENS)
    meta = {
        "model": args.model,
        "paraphrases": str(PARAPHRASES_PATH),
        "dim": int(keys.shape[1]),
        "label_source": "openai",
        "input_text_vecs": "Qwen3-0.6B",
        **paraphrase_stats,
        "api_tokens": int(api_tokens),
        "approx_cost_usd": round(api_tokens / 1e6 * usd_per_million_tokens, 2),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--out", dest="output_dir", default=str(paths.DATA / "keys_openai"))
    parser.add_argument(
        "--para_keep_frac",
        type=float,
        default=0.927,
        help="drift filter: keep this fraction of paraphrases (same fraction as the Qwen 0.6B key set)",
    )
    parser.add_argument("--batch", type=int, default=256, help="inputs per API request")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def embed_each(client, args: argparse.Namespace, texts: list[str], cache_prefix: Path) -> tuple[np.ndarray, int]:
    """One embedding row per text (each distinct text is sent once) and the API tokens spent."""
    unique_texts, positions = key_set.deduplicate(texts)
    vectors, api_tokens = embed_cached(client, args.model, unique_texts, args.batch, args.workers, cache_prefix)
    return vectors[positions], api_tokens


def embed_cached(
    client, model: str, texts: list[str], batch_size: int, workers: int, cache_prefix: Path
) -> tuple[np.ndarray, int]:
    """L2-normalized embeddings of texts and the API tokens spent.

    <cache_prefix>.npy / .json hold the vectors and their texts; they are reused (0 tokens) if the texts match."""
    vectors_path, texts_path = Path(f"{cache_prefix}.npy"), Path(f"{cache_prefix}.json")
    if vectors_path.exists() and texts_path.exists() and json.loads(texts_path.read_text()) == texts:
        print(f"cached: {vectors_path}", flush=True)
        return np.load(vectors_path), 0
    chunks = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]

    def embed_chunk(chunk: list[str]) -> tuple[np.ndarray, int]:
        # the API rejects empty strings
        inputs = [text[:MAX_INPUT_CHARACTERS] if text.strip() else "." for text in chunk]
        response = client.embeddings.create(model=model, input=inputs)
        in_input_order = sorted(response.data, key=lambda item: item.index)
        return np.array([item.embedding for item in in_input_order], np.float32), response.usage.total_tokens

    chunk_vectors, api_tokens, start_time = [], 0, time.time()
    with ThreadPoolExecutor(workers) as executor:
        for vectors, tokens in executor.map(embed_chunk, chunks):
            chunk_vectors.append(vectors)
            api_tokens += tokens
            if len(chunk_vectors) % 50 == 0 or len(chunk_vectors) == len(chunks):
                print(
                    f"  {len(chunk_vectors)}/{len(chunks)} requests  {api_tokens / 1e6:.2f}M tokens  "
                    f"{time.time() - start_time:.0f}s",
                    flush=True,
                )
    vectors = np.concatenate(chunk_vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(vectors_path, vectors)
    texts_path.write_text(json.dumps(texts))
    return vectors, api_tokens


def link_typed_text_vectors(output_dir: Path) -> None:
    """Typed-text vectors are model inputs, not labels: every key set uses the default set's text_vecs.npy."""
    link = output_dir / "text_vecs.npy"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(paths.SHARED / "text_vecs.npy")


if __name__ == "__main__":
    main()
