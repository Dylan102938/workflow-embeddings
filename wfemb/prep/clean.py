"""The rules a session of an added dataset must pass to become training data. Their reasons: docs/DATA_PREP.md."""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

import numpy as np

from .actions import Event
from .held_out import HeldOut

MIN_STEPS, MAX_STEPS = 2, 100
MIN_INSTRUCTION_WORDS, MAX_INSTRUCTION_WORDS = 2, 150
MIN_ACTING_STEP_FRACTION = 0.5
MAX_EDGE_POINTER_FRACTION = 0.3
PORTRAIT_RATIO = 1.25  # height / width above which a screenshot is a phone screen
BLANK_SPREAD = 0.5  # grey-level standard deviation below which a screenshot shows nothing (dark themes: 1 to 2)
MAX_BLANK_FRACTION = 0.2
MAX_SESSIONS_PER_INSTRUCTION = 8
SIGNATURE_GRID = 8
PASSIVE_EVENT_TYPES = {"other", "wait"}


@dataclass
class Candidate:
    id: str
    instruction: str
    group: str
    applications: list[str]  # application and website names
    steps: list[dict]  # {image, events, focus}
    screenshots: list[dict | None]  # per step {size, digest, spread}; None: the file is missing or does not decode

    @property
    def screenshot_files(self) -> tuple:
        return tuple(stats and stats["digest"] for stats in self.screenshots)

    @property
    def actions(self) -> tuple:
        return tuple(tuple(event_signature(event) for event in step["events"]) for step in self.steps)


@dataclass
class Kept:
    """The sessions kept so far."""

    ids: set = field(default_factory=set)
    screenshot_sequences: set = field(default_factory=set)
    demonstrations: set = field(default_factory=set)
    per_instruction: collections.Counter = field(default_factory=collections.Counter)

    def add(self, session: Candidate) -> None:
        self.ids.add(session.id)
        self.screenshot_sequences.add(session.screenshot_files)
        self.demonstrations.add((session.group, session.actions))
        self.per_instruction[session.group] += 1


# A session is dropped by the first rule that is true for it.
HELD_OUT_RULES = {
    "held_out_application": lambda session, held_out: held_out.has_application(session.applications),
    "mentions_held_out_application": lambda session, held_out: held_out.is_named_in(session.instruction),
    "evaluation_instruction": lambda session, held_out: session.group in held_out.instructions,
    "near_evaluation_instruction": lambda session, held_out: held_out.has_near_copy_of(session.instruction),
}
QUALITY_RULES = {
    "bad_instruction": lambda session: (
        not MIN_INSTRUCTION_WORDS <= len(session.instruction.split()) <= MAX_INSTRUCTION_WORDS
        or not re.search(r"[^\W\d_]", session.instruction)
    ),
    "missing_screenshot": lambda session: None in session.screenshots,
    "too_short": lambda session: len(session.steps) < MIN_STEPS,
    "too_long": lambda session: len(session.steps) > MAX_STEPS,
    "portrait_screen": lambda session: any(
        stats["size"][1] > PORTRAIT_RATIO * stats["size"][0] for stats in session.screenshots
    ),
    "few_input_actions": lambda session: acting_step_fraction(session.steps) < MIN_ACTING_STEP_FRACTION,
    "pointer_off_screen": lambda session: edge_pointer_fraction(session.steps) > MAX_EDGE_POINTER_FRACTION,
    "blank_screenshots": lambda session: (
        is_blank(session.screenshots[0]) or np.mean(list(map(is_blank, session.screenshots))) > MAX_BLANK_FRACTION
    ),
    "identical_screenshots": lambda session: len(set(session.screenshot_files)) == 1,
}
DUPLICATE_RULES = {
    "duplicate_id": lambda session, kept: session.id in kept.ids,
    "duplicate_screenshots": lambda session, kept: session.screenshot_files in kept.screenshot_sequences,
    "duplicate_demonstration": lambda session, kept: (session.group, session.actions) in kept.demonstrations,
    "instruction_repeated_too_often": lambda session, kept: (
        kept.per_instruction[session.group] >= MAX_SESSIONS_PER_INSTRUCTION
    ),
}


RULES = (HELD_OUT_RULES, QUALITY_RULES, DUPLICATE_RULES)


def rejection_reason(session: Candidate, held_out: HeldOut, kept: Kept) -> str | None:
    """The first rule the session breaks, or None to keep it."""
    for rules, arguments in zip(RULES, ((session, held_out), (session,), (session, kept))):
        for name, breaks in rules.items():
            if breaks(*arguments):
                return name
    return None


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


def is_blank(screenshot: dict) -> bool:
    return screenshot["spread"] < BLANK_SPREAD


def event_signature(event: Event) -> tuple:
    """An event with its positions rounded to a coarse grid: equal for two recordings of the same actions."""
    cells = tuple(
        None if value is None else int(value * SIGNATURE_GRID) for value in (event.x, event.y, event.x2, event.y2)
    )
    return (event.type, cells, tuple(event.keys), event.text)
