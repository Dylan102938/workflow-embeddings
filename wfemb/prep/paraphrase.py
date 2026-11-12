"""Paraphrases every unique instruction with a local instruction-tuned LLM (vLLM); writes keys/paraphrases.json.

  python -m wfemb.prep.paraphrase [--model Qwen/Qwen2.5-7B-Instruct] [--n 6] [--temperature 0.9] [--limit N] [--tp 1]

Runs in its own environment (runners/slurm/paraphrase.sbatch): vLLM pins its own torch, so this module and what
it imports may only need the standard library, numpy and vllm.
"""

from __future__ import annotations

import argparse
import json
import re

from ..storage import paths
from .prepare import load_sessions

PARAPHRASES_PATH = paths.SHARED / "paraphrases.json"

PROMPT = """Rewrite the computer-task instruction below in {n} different ways.

Rules:
- Every rewrite must ask for exactly the same task, with the same goal and outcome.
- Vary the wording and sentence structure (imperative, question, polite request, terse note, etc.).
- Keep application names, website names and concrete values (file names, numbers, people, text to type) unchanged.
- Do not add, remove or reorder requirements.

Return only a JSON array of {n} strings.

Instruction: {instruction}"""


def load_paraphrases() -> dict[str, list[str]]:
    """Instruction -> its paraphrases, unfiltered (the key-set builders apply the drift filter)."""
    with open(PARAPHRASES_PATH) as f:
        return json.load(f)["paraphrases"]


def main() -> None:
    args = parse_arguments()
    from vllm import LLM, SamplingParams

    instructions = sorted({session["instruction"] for session in load_sessions()})
    if args.limit:
        instructions = instructions[: args.limit]
    print(f"{len(instructions)} unique instructions")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        seed=0,
    )
    sampling = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=1024, seed=0)
    conversations = [
        [{"role": "user", "content": PROMPT.format(n=args.count, instruction=instruction)}]
        for instruction in instructions
    ]
    replies = [output.outputs[0].text for output in llm.chat(conversations, sampling, use_tqdm=True)]

    paraphrases = {
        instruction: distinct_paraphrases(reply, instruction, args.count)
        for instruction, reply in zip(instructions, replies)
    }
    paths.SHARED.mkdir(parents=True, exist_ok=True)
    with open(PARAPHRASES_PATH, "w") as f:
        json.dump(
            {"model": args.model, "n": args.count, "paraphrases": paraphrases, "raw": dict(zip(instructions, replies))},
            f,
        )

    incomplete = sum(len(rewrites) < args.count for rewrites in paraphrases.values())
    print(f"wrote {PARAPHRASES_PATH}; {incomplete} instructions got fewer than {args.count} paraphrases")
    for instruction in instructions[:5]:
        print("\n#", instruction)
        for rewrite in paraphrases[instruction]:
            print("  -", rewrite)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", dest="count", type=int, default=6, help="paraphrases per instruction")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="only the first N instructions")
    parser.add_argument("--tp", dest="tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def distinct_paraphrases(reply: str, instruction: str, count: int) -> list[str]:
    """The paraphrases in an LLM reply, without copies of the instruction itself."""
    return [rewrite for rewrite in parse_paraphrases(reply, count) if rewrite.lower() != instruction.lower()]


def parse_paraphrases(reply: str, limit: int) -> list[str]:
    """The strings of the JSON array in the reply; if it is not one array, one paraphrase per line.

    A line may be numbered or bulleted, or an array of its own: ["Please resize the clip."]"""
    array = re.search(r"\[.*\]", reply, flags=re.S)
    if array:
        try:
            items = json.loads(array.group(0))
            return [str(item).strip() for item in items if isinstance(item, str) and item.strip()][:limit]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip(" \t[],").strip('"') for line in reply.splitlines()]
    return [line for line in lines if len(line) > 3][:limit]


if __name__ == "__main__":
    main()
