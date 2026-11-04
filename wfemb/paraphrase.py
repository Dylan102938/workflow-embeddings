"""Step 1c' (offline, GPU): paraphrase every instruction with a local instruction-tuned LLM (vLLM).

Same intent, different wording; app/site names and concrete values are kept. The drift filter
(THETA_PARA, cosine to the original key) is applied later in text_embed.py.
Runs in its own environment (vLLM pins its own torch):  see scripts/paraphrase.sbatch
"""
from __future__ import annotations

import argparse
import json
import pickle
import re

from . import paths

PROMPT = """Rewrite the computer-task instruction below in {n} different ways.

Rules:
- Every rewrite must ask for exactly the same task, with the same goal and outcome.
- Vary the wording and sentence structure (imperative, question, polite request, terse note, etc.).
- Keep application names, website names and concrete values (file names, numbers, people, text to type) unchanged.
- Do not add, remove or reorder requirements.

Return only a JSON array of {n} strings.

Instruction: {instruction}"""


def parse_list(text: str, n: int) -> list[str]:
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            return [str(x).strip() for x in out if isinstance(x, str) and x.strip()][:n]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r'^\s*(\d+[.)]|[-*])\s*', "", l).strip().strip('",') for l in text.splitlines()]
    return [l for l in lines if len(l) > 3 and not l.startswith(("[", "]"))][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    instructions = sorted({s["instruction"] for s in sessions})
    if args.limit:
        instructions = instructions[: args.limit]
    print(f"{len(instructions)} unique instructions")

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=1024, seed=0)
    msgs = [[{"role": "user", "content": PROMPT.format(n=args.n, instruction=ins)}] for ins in instructions]
    outs = llm.chat(msgs, sp, use_tqdm=True)

    result, short = {}, 0
    for ins, o in zip(instructions, outs):
        paras = [p for p in parse_list(o.outputs[0].text, args.n) if p.lower() != ins.lower()]
        short += len(paras) < args.n
        result[ins] = paras
    paths.KEYS.mkdir(parents=True, exist_ok=True)
    out_path = paths.KEYS / "paraphrases.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "n": args.n, "paraphrases": result}, f)
    print(f"wrote {out_path}; {short} instructions got fewer than {args.n} paraphrases")
    for ins in instructions[:5]:
        print("\n#", ins)
        for p in result[ins]:
            print("  -", p)


if __name__ == "__main__":
    main()
