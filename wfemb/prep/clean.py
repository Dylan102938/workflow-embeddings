"""Which sessions of an added dataset are fit for training. Rules and their reasons: docs/DATA_PREP.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .actions import Event

MIN_STEPS = 2
MAX_STEPS = 100
MIN_INSTRUCTION_WORDS = 2
MAX_INSTRUCTION_WORDS = 150
MIN_ACTING_STEP_FRACTION = 0.5
BLANK_SPREAD = 2.0  # grey-level standard deviation below which a screenshot shows nothing
MAX_BLANK_FRACTION = 0.2
MAX_EDGE_POINTER_FRACTION = 0.3
EVALUATION_SIMILARITY = 0.9  # character n-gram TF-IDF cosine to a validation or test instruction
MAX_SESSIONS_PER_INSTRUCTION = 8
SIGNATURE_GRID = 8

PASSIVE_EVENT_TYPES = {"other", "wait"}
VENDOR_WORDS = {"mozilla", "microsoft", "google", "apple", "adobe", "mac", "macos", "windows", "gnu"}
NAME_SUFFIXES = ("community edition", "media player", "file manager", "browser", "ide", "app", "desktop")
# Held-out names that are also ordinary words or part of other applications' names: not matched in instruction text.
AMBIGUOUS_NAMES = {
    "arc",
    "brave",
    "calendar",
    "everything",
    "keynote",
    "libreoffice",
    "loom",
    "maps",
    "meta",
    "monday",
    "numbers",
    "pages",
    "spring",
    "steam",
    "zoom",
}
MIN_MENTION_LENGTH = 4


@dataclass
class Candidate:
    id: str
    instruction: str
    group: str
    applications: list[str]  # application and website names
    steps: list[dict]  # {image, events, focus}
    image_stats: list[dict | None]  # per step; None: the screenshot is missing or does not decode
    evaluation_similarity: float = 0.0
    signature: tuple = field(init=False)

    def __post_init__(self):
        self.signature = tuple(tuple(event_signature(event) for event in step["events"]) for step in self.steps)


@dataclass
class EvaluationSets:
    """What the validation and test splits hold out, which training data must not contain."""

    groups: set[str]
    application_names: set[str]  # canonical
    mentioned_names: re.Pattern


class Deduplicator:
    """Remembers the sessions accepted so far."""

    def __init__(self):
        self.ids: set[str] = set()
        self.screenshot_sequences: set[tuple] = set()
        self.demonstrations: set[tuple] = set()
        self.sessions_per_group: dict[str, int] = {}

    def accept(self, candidate: Candidate) -> None:
        self.ids.add(candidate.id)
        self.screenshot_sequences.add(screenshot_sequence(candidate))
        self.demonstrations.add((candidate.group, candidate.signature))
        self.sessions_per_group[candidate.group] = self.sessions_per_group.get(candidate.group, 0) + 1


def rejection_reason(candidate: Candidate, held_out: EvaluationSets, seen: Deduplicator) -> str | None:
    """The first rule the session breaks, or None to keep it."""
    if candidate.id in seen.ids:
        return "duplicate_id"
    words = candidate.instruction.split()
    if not MIN_INSTRUCTION_WORDS <= len(words) <= MAX_INSTRUCTION_WORDS or not re.search(
        r"[^\W\d_]", candidate.instruction
    ):
        return "bad_instruction"
    if any(canonical_application(name) in held_out.application_names for name in candidate.applications):
        return "held_out_application"
    if held_out.mentioned_names.search(candidate.instruction):
        return "mentions_held_out_application"
    if candidate.group in held_out.groups:
        return "evaluation_instruction"
    if candidate.evaluation_similarity >= EVALUATION_SIMILARITY:
        return "near_evaluation_instruction"
    if any(stats is None for stats in candidate.image_stats):
        return "missing_screenshot"
    if len(candidate.steps) < MIN_STEPS:
        return "too_short"
    if len(candidate.steps) > MAX_STEPS:
        return "too_long"
    if any(stats["size"][0] < stats["size"][1] for stats in candidate.image_stats):
        return "portrait_screen"
    if acting_step_fraction(candidate.steps) < MIN_ACTING_STEP_FRACTION:
        return "few_input_actions"
    if edge_pointer_fraction(candidate.steps) > MAX_EDGE_POINTER_FRACTION:
        return "pointer_off_screen"
    blank = [stats["spread"] < BLANK_SPREAD for stats in candidate.image_stats]
    if blank[0] or np.mean(blank) > MAX_BLANK_FRACTION:
        return "blank_screenshots"
    if len(set(screenshot_sequence(candidate))) == 1:
        return "identical_screenshots"
    if screenshot_sequence(candidate) in seen.screenshot_sequences:
        return "duplicate_screenshots"
    if (candidate.group, candidate.signature) in seen.demonstrations:
        return "duplicate_demonstration"
    if seen.sessions_per_group.get(candidate.group, 0) >= MAX_SESSIONS_PER_INSTRUCTION:
        return "instruction_repeated_too_often"
    return None


def event_signature(event: Event) -> tuple:
    """An event with its positions rounded to a coarse grid: equal for two recordings of the same actions."""
    cells = tuple(
        None if value is None else int(value * SIGNATURE_GRID) for value in (event.x, event.y, event.x2, event.y2)
    )
    return (event.type, cells, tuple(event.keys), event.text)


def screenshot_sequence(candidate: Candidate) -> tuple:
    """The files of the session's screenshots: equal for a recording that is listed twice."""
    return tuple(stats["digest"] if stats else None for stats in candidate.image_stats)


def acting_step_fraction(steps: list[dict]) -> float:
    """Of the steps before the final screenshot, the fraction with a pointer or keyboard event."""
    acting = [any(event.type not in PASSIVE_EVENT_TYPES for event in step["events"]) for step in steps]
    if not acting[-1]:
        acting = acting[:-1]
    return float(np.mean(acting)) if acting else 0.0


def edge_pointer_fraction(steps: list[dict]) -> float:
    """Positions exactly on the screen border are clipped ones: the source's coordinates were off the screenshot."""
    positions = [
        value for step in steps for event in step["events"] for value in (event.x, event.y) if value is not None
    ]
    return float(np.mean([value <= 0.0 or value >= 1.0 for value in positions])) if positions else 0.0


# ------------------------------------------------------------------------------------------- what is held out
def canonical_application(name: str) -> str:
    """'Mozilla Thunderbird', 'thunderbird', 'domain:thunderbird' -> 'thunderbird'."""
    name = re.sub(r"[_\s]+", " ", name.lower().split(":", 1)[-1]).strip()
    for suffix in NAME_SUFFIXES:
        name = name.removesuffix(f" {suffix}")
    words = name.split()
    while len(words) > 1 and words[0] in VENDOR_WORDS:
        words = words[1:]
    return " ".join(words)


def evaluation_sets(sessions: list[dict], test_applications: list[str]) -> EvaluationSets:
    names = [re.sub(r"[_\s]+", " ", key.split(":", 1)[1].lower()).strip() for key in test_applications]
    mentionable = sorted(
        {name for name in names if len(name) >= MIN_MENTION_LENGTH and name not in AMBIGUOUS_NAMES}, key=len
    )
    pattern = r"(?<![\w.])(?:" + "|".join(re.escape(name) for name in reversed(mentionable)) + r")(?![\w])"
    return EvaluationSets(
        groups={session["group"] for session in sessions if session["split"] != "train"},
        application_names={canonical_application(key) for key in test_applications},
        mentioned_names=re.compile(pattern, re.IGNORECASE),
    )


def similarity_to_evaluation(instructions: list[str], evaluation_instructions: list[str]) -> np.ndarray:
    """For every instruction, the largest character n-gram TF-IDF cosine to a validation or test instruction."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
    evaluation = vectorizer.fit_transform(sorted(set(evaluation_instructions)))
    largest = np.zeros(len(instructions), np.float32)
    for start in range(0, len(instructions), 2048):
        similarities = vectorizer.transform(instructions[start : start + 2048]) @ evaluation.T
        largest[start : start + 2048] = similarities.max(axis=1).toarray().ravel()
    return largest
