"""Builds the bundles that the dashboard's Demo tab plays. What a bundle is and where its data comes from: docs/DEMO.md.

python -m wfemb.demo bundle --id <run id>        # background + export --auto 8 + frames
python -m wfemb.demo background --id <run id>    # the map of all test workflows
python -m wfemb.demo export --id <run id> --auto 8 | --session <session id> | --raw workflow.json
python -m wfemb.demo frames --id <run id>        # screenshots of the bundle's workflows
python -m wfemb.demo screens --screens <folder>  # once, where the AgentNet archives are: the screenshot cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import REPO, Experiment
from .assets import DemoAssets
from .background import BackgroundMap, write_background
from .raw import export_raw_workflow
from .screenshots import DEFAULT_CACHE, cache_test_screenshots, write_bundle_frames
from .selection import choose_illustrative, judged_partners
from .workflow import workflow_of_test_session, write_workflow

DEFAULT_AUTO = 8  # illustrative workflows in a bundle


def bundle(arguments: argparse.Namespace):
    """background, export (at least the --auto workflows) and frames."""
    assets = load_assets(arguments)
    background(arguments, assets)
    arguments.auto = arguments.auto or DEFAULT_AUTO
    arguments.session = arguments.session + like_bundle_sessions(Path(arguments.out), arguments.like)
    export(arguments, assets)
    frames(arguments)


def like_bundle_sessions(bundle_dir: Path, like_run_id: str) -> list[str]:
    """The test sessions in the --like run's bundle (the folder next to this bundle): exporting them here as well
    lets two runs be compared on the same workflows."""
    if not like_run_id:
        return []
    like_dir = bundle_dir.parent / Experiment.from_id(like_run_id).id
    index_path = like_dir / "workflows" / "index.json"
    if like_dir == bundle_dir or not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return [entry["id"] for entry in index if entry.get("source") == "agentnet-test"]


def background(arguments: argparse.Namespace, assets: DemoAssets | None = None):
    write_background(assets or load_assets(arguments), Path(arguments.out), arguments.k)


def export(arguments: argparse.Namespace, assets: DemoAssets | None = None):
    assets = assets or load_assets(arguments)
    bundle_dir = Path(arguments.out)
    background_map = BackgroundMap.load(assets.run_id)
    partners = judged_partners(assets, background_map)

    session_indices = []
    if arguments.auto:
        positions = choose_illustrative(assets, background_map, partners, arguments.auto, arguments.hard)
        session_indices += [int(background_map.session_indices[position]) for position in positions]
    for session_id in arguments.session:
        session_index = assets.session_index_of.get(session_id)
        if session_index is None or assets.splits[session_index] != "test":
            raise SystemExit(
                f"{session_id}: not a test session (use --raw with its screenshots and code for other sessions)"
            )
        session_indices.append(session_index)

    for session_index in session_indices:
        workflow = workflow_of_test_session(assets, background_map, partners, session_index, arguments.topk)
        write_workflow(bundle_dir, workflow)
        steps = workflow["steps"]
        print(
            f"{workflow['app'][:24]:24s} {workflow['os']:8s} {len(steps):3d} steps  "
            f"true rank {steps[0]['true_rank']:>5} -> {steps[-1]['true_rank']:<4} "
            f"partners {len(workflow['partners'])}  {workflow['instruction'][:70]}",
            flush=True,
        )

    for raw_path in arguments.raw:
        workflow = export_raw_workflow(
            assets,
            background_map,
            Path(raw_path),
            bundle_dir,
            images_dir=Path(arguments.images) if arguments.images else None,
            channel_pca_path=Path(arguments.frames_pca) if arguments.frames_pca else None,
            single_file=arguments.single_file,
            top_k=arguments.topk,
        )
        path = write_workflow(bundle_dir, workflow, arguments.single_file)
        steps = workflow["steps"]
        print(
            f"raw workflow {workflow['id']}: {len(steps)} steps -> {path}; "
            f"final top intent: {steps[-1]['top'][0][0][:90]}"
        )


def frames(arguments: argparse.Namespace):
    write_bundle_frames(Path(arguments.out), screenshot_cache(arguments))


def screens(arguments: argparse.Namespace):
    if not cache_test_screenshots(screenshot_cache(arguments), arguments.workers):
        raise SystemExit("no screenshot could be written")


def load_assets(arguments: argparse.Namespace) -> DemoAssets:
    return DemoAssets(arguments.id, arguments.ckpt)


def screenshot_cache(arguments: argparse.Namespace) -> Path:
    return Path(arguments.screens) if arguments.screens else DEFAULT_CACHE


COMMANDS = {"bundle": bundle, "background": background, "export": export, "frames": frames, "screens": screens}


def main():
    best_run_id = Experiment().id
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--id", default=best_run_id, help="run id (default: the best model's)")
    parser.add_argument("--ckpt", default="best.pt", help="export --raw: checkpoint in the run's results folder")
    parser.add_argument("--out", default="", help="bundle folder (default dashboard/bundles/<run id>)")
    parser.add_argument(
        "--like",
        default=best_run_id,
        help="bundle: also export the test workflows of this run's bundle (default: the best model's)",
    )
    parser.add_argument("--screens", default="", help=f"screenshot cache (default {DEFAULT_CACHE})")
    parser.add_argument("--workers", type=int, default=16, help="screens: threads")
    parser.add_argument("--k", type=int, default=10, help="background: number of clusters")
    parser.add_argument("--auto", type=int, default=0, help="export: number of illustrative test workflows to choose")
    parser.add_argument("--hard", type=int, default=2, help="export --auto: how many of them are hard cases")
    parser.add_argument("--session", action="append", default=[], help="export: AgentNet test session id")
    parser.add_argument("--raw", action="append", default=[], help="export: raw workflow JSON (docs/DEMO.md)")
    parser.add_argument("--images", default="", help="export --raw: folder of the screenshots")
    parser.add_argument("--frames_pca", default="", help="export --raw: channel-PCA basis (default: in WFEMB_BULK)")
    parser.add_argument("--single_file", action="store_true", help="export --raw: embed screenshots in the JSON")
    parser.add_argument("--topk", type=int, default=5, help="export: intents listed per step")
    arguments = parser.parse_args()
    arguments.id = Experiment.from_id(arguments.id).id
    arguments.out = arguments.out or str(REPO / "dashboard" / "bundles" / arguments.id)
    COMMANDS[arguments.command](arguments)


if __name__ == "__main__":
    main()
