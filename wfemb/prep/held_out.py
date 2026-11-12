"""What AgentNet's validation and test splits hold out from training data, and the check that training data has none.

  python -m wfemb.prep.held_out

Prints every rule for the test set with the number of sessions that break it, and exits with 1 if any does.
The rules and their reasons: docs/DATA_PREP.md, "Rules for the test set".
"""

from __future__ import annotations

import collections
import json
import pickle
import re
import sys
from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from ..storage import paths

NEAR_COPY_SIMILARITY = 0.9  # cosine of character 3-to-5-gram TF-IDF vectors
MIN_MENTION_LENGTH = 4
VENDOR_WORDS = {"mozilla", "microsoft", "google", "apple", "adobe", "mac", "macos", "windows", "gnu"}
NAME_SUFFIXES = ("community edition", "media player", "file manager", "browser", "ide", "app", "desktop")
# Held-out names that are also ordinary words or part of other applications' names: not looked for in instructions.
AMBIGUOUS_NAMES = set(
    "arc brave calendar everything keynote libreoffice loom maps meta monday numbers pages spring steam zoom".split()
)


def canonical_application(name: str) -> str:
    """'Mozilla Thunderbird', 'thunderbird', 'domain:thunderbird' -> 'thunderbird'."""
    name = re.sub(r"[_\s]+", " ", name.lower().split(":", 1)[-1]).strip()
    for suffix in NAME_SUFFIXES:
        name = name.removesuffix(f" {suffix}")
    words = name.split()
    while len(words) > 1 and words[0] in VENDOR_WORDS:
        words = words[1:]
    return " ".join(words)


class HeldOut:
    """What no session of an added dataset may contain."""

    def __init__(self, agentnet_sessions: list[dict], test_applications: list[str]):
        training = [session for session in agentnet_sessions if session["split"] == "train"]
        evaluation = [session for session in agentnet_sessions if session["split"] != "train"]
        training_names = {canonical_application(key) for session in training for key in session["app_keys"]}
        evaluation_names = {canonical_application(key) for session in evaluation for key in session["app_keys"]}
        # the test applications, and every other application that only validation or test sessions use
        self.applications = {canonical_application(key) for key in test_applications} | (
            evaluation_names - training_names
        )
        self.instructions = {session["group"] for session in evaluation}

        names = {re.sub(r"[_\s]+", " ", key.split(":", 1)[1].lower()).strip() for key in test_applications}
        mentionable = sorted(name for name in names if len(name) >= MIN_MENTION_LENGTH and name not in AMBIGUOUS_NAMES)
        self._mention = re.compile(r"(?<![\w.@])(?:" + "|".join(map(re.escape, mentionable)) + r")(?!\w)", re.I)
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
        self._instruction_vectors = self._vectorizer.fit_transform(
            sorted({session["instruction"] for session in evaluation})
        ).T.tocsr()

    def has_application(self, names: Iterable[str]) -> bool:
        return any(canonical_application(name) in self.applications for name in names)

    def is_named_in(self, instruction: str) -> bool:
        return bool(self._mention.search(instruction))

    def has_near_copy_of(self, instruction: str) -> bool:
        similarities = self._vectorizer.transform([instruction]) @ self._instruction_vectors
        return similarities.nnz > 0 and similarities.max() >= NEAR_COPY_SIMILARITY


def violations(agentnet: list[dict], added: list[dict], test_applications: list[str]) -> dict[str, list[str]]:
    """Every rule for the test set, with the ids of the sessions that break it."""
    held_out = HeldOut(agentnet, test_applications)
    test_keys = set(test_applications)

    def has_test_application(session: dict) -> bool:
        return bool(test_keys & {session["app"], *session["app_keys"]})

    test = [session for session in agentnet if session["split"] == "test"]
    test_instructions = {session["group"] for session in test if has_test_application(session)}
    splits_of_instruction = collections.defaultdict(set)
    for session in agentnet:
        splits_of_instruction[session["group"]].add(session["split"])

    rules = {
        "AgentNet: every test session has a test application, or the instruction of a session that has one": [
            session
            for session in test
            if not has_test_application(session) and session["group"] not in test_instructions
        ],
        "AgentNet: no training or validation session has a test application": [
            session for session in agentnet if session["split"] != "test" and has_test_application(session)
        ],
        "AgentNet: no instruction is in two splits": [
            session for session in agentnet if len(splits_of_instruction[session["group"]]) > 1
        ],
        "added: every session is training data": [session for session in added if session["split"] != "train"],
        "added: no session has a held-out application": [
            session for session in added if held_out.has_application(session["app_keys"])
        ],
        "added: no instruction names a held-out application": [
            session for session in added if held_out.is_named_in(session["instruction"])
        ],
        "added: no instruction is a validation or test instruction, or a near copy of one": [
            session
            for session in added
            if session["group"] in held_out.instructions or held_out.has_near_copy_of(session["instruction"])
        ],
    }
    return {rule: [session["id"] for session in sessions] for rule, sessions in rules.items()}


def main() -> None:
    with open(paths.AGENTNET_DATA / "prepared" / "sessions.pkl", "rb") as f:
        agentnet = pickle.load(f)
    with open(paths.ADDED_DATA / "prepared" / "sessions.pkl", "rb") as f:
        added = pickle.load(f)
    test_applications = json.load(open(paths.AGENTNET_DATA / "prepared" / "summary.json"))["test_apps"]
    broken = violations(agentnet, added, test_applications)
    for rule, session_ids in broken.items():
        print(f"{'ok' if not session_ids else f'BROKEN by {len(session_ids)}: {session_ids[:5]}':<12} {rule}")
    names = {canonical_application(key) for key in test_applications}
    same_name = sum(
        session["split"] == "train" and not names.isdisjoint(map(canonical_application, session["app_keys"]))
        for session in agentnet
    )
    print(f"not a rule: {same_name} AgentNet training sessions have a test application's name under another key")
    sys.exit(1 if any(broken.values()) else 0)


if __name__ == "__main__":
    main()
