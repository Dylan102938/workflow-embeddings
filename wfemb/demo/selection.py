"""Which test workflows a bundle shows, and which other test workflows each one is linked to."""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

from ..storage import paths
from .assets import DemoAssets
from .background import BackgroundMap

SAME_WORKFLOW = 2  # judged labels: 2 = same workflow, 1 = related, 0 = different
MIN_STEPS, MAX_STEPS = 8, 22  # long enough to show the belief change, short enough to scrub through
MAX_INSTRUCTION_LENGTH = 150
SOLVED_RANK = 3  # an illustrative workflow ends with the true intent at this rank or better
UNDECIDED_RANK = 5  # ... and starts with it worse than this
HARD_RANK = 30  # a hard case ends with the true intent worse than this


class Partners(NamedTuple):
    """Positions on the background map of the test workflows linked to one workflow."""

    same_workflow: list[int]
    related: list[int]


def judged_partners(assets: DemoAssets, background_map: BackgroundMap) -> list[Partners]:
    """Per map position: the workflows judged the same (or in the same group), and those judged related."""
    session_indices = background_map.session_indices
    position_of = {int(session_index): position for position, session_index in enumerate(session_indices)}
    same: dict[int, set[int]] = {}
    related: dict[int, set[int]] = {}

    def link(links: dict[int, set[int]], a: int, b: int):
        links.setdefault(a, set()).add(b)
        links.setdefault(b, set()).add(a)

    for line in open(paths.SHARED / "judged_test.jsonl"):
        pair = json.loads(line)
        a, b, label = position_of.get(pair["a"]), position_of.get(pair["b"]), int(pair["label"])
        if a is not None and b is not None and label != 0:
            link(same if label == SAME_WORKFLOW else related, a, b)

    positions_in_group: dict[str, list[int]] = {}
    for position, session_index in enumerate(session_indices):
        positions_in_group.setdefault(assets.sessions[session_index]["group"], []).append(position)
    for positions in positions_in_group.values():
        for a in positions:
            for b in positions:
                if a != b:
                    link(same, a, b)

    return [
        Partners(sorted(same.get(position, set())), sorted(related.get(position, set()) - same.get(position, set())))
        for position in range(len(session_indices))
    ]


def choose_illustrative(
    assets: DemoAssets,
    background_map: BackgroundMap,
    partners: list[Partners],
    n_workflows: int,
    n_hard: int,
    seed: int = 0,
) -> list[int]:
    """Map positions of n_workflows test workflows worth showing, n_hard of them hard cases.

    The others are workflows whose belief sharpens onto the true intent, one per app, balanced over operating
    systems, preferring a large entropy drop and workflows that have a same-workflow partner.
    """
    rng = np.random.default_rng(seed)
    sessions = [assets.sessions[index] for index in background_map.session_indices]
    first_ranks, last_ranks = background_map.first_step_ranks, background_map.last_step_ranks
    entropy_drops = background_map.first_step_entropies - background_map.last_step_entropies
    n_steps = assets.events["sess_len"][background_map.session_indices]
    has_partner = np.array([len(partner.same_workflow) > 0 for partner in partners])
    showable = (
        (n_steps >= MIN_STEPS)
        & (n_steps <= MAX_STEPS)
        & np.array([len(session["instruction"]) <= MAX_INSTRUCTION_LENGTH for session in sessions])
    )

    candidates = np.nonzero(showable & (last_ranks <= SOLVED_RANK) & (first_ranks > UNDECIDED_RANK))[0]
    preference = entropy_drops[candidates] + 1.5 * has_partner[candidates] + 0.1 * rng.random(len(candidates))
    candidates = candidates[np.argsort(-preference)]
    hard_candidates = np.nonzero(showable & (last_ranks > HARD_RANK))[0]
    hard_preference = has_partner[hard_candidates] + 0.1 * rng.random(len(hard_candidates))
    hard_candidates = hard_candidates[np.argsort(-hard_preference)]

    chosen: list[int] = []
    apps_shown: set[str] = set()
    n_per_os: dict[str, int] = {}
    n_sharpening = n_workflows - n_hard
    # The second pass drops the per-OS limit, for when an operating system runs out of candidates.
    for os_limit in (max(2, (n_sharpening + 2) // 3), 10**9):
        for position in candidates:
            app, os_name = sessions[position]["app"], sessions[position]["os"]
            if len(chosen) >= n_sharpening or app in apps_shown or n_per_os.get(os_name, 0) >= os_limit:
                continue
            chosen.append(int(position))
            apps_shown.add(app)
            n_per_os[os_name] = n_per_os.get(os_name, 0) + 1

    n_hard_chosen = 0
    for position in hard_candidates:
        if n_hard_chosen >= n_hard:
            break
        app = sessions[position]["app"]
        if app not in apps_shown:
            chosen.append(int(position))
            apps_shown.add(app)
            n_hard_chosen += 1
    return chosen
