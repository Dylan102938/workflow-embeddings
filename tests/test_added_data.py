"""Converters, cleaning rules, rules for the test set and loading of the datasets added to AgentNet."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import add_synthetic_datasets  # noqa: E402
from make_synthetic import main as make_synthetic  # noqa: E402

from wfemb.config import DataConfig, EvalConfig, Experiment, RunConfig, TrainConfig  # noqa: E402
from wfemb.prep import clean  # noqa: E402
from wfemb.prep.actions import parse_code  # noqa: E402
from wfemb.prep.held_out import HeldOut, violations  # noqa: E402
from wfemb.prep.prepare import group_hash  # noqa: E402
from wfemb.prep.sources import gui360, scalecua, videocua  # noqa: E402
from wfemb.prep.sources.common import is_key_combination, key_tokens  # noqa: E402

SMALL = dict(d_model=64, d_head=16, n_heads=4, n_layers=2, n_patches=16, batch_size=16, n_epochs=1, learning_rate=1e-3)


# -------------------------------------------------------------------------------------------------- converters
def test_key_names_follow_agentnet():
    names = {"Ctrl + Shift+I": ["ctrl", "shift", "i"], '"Cmd" + "V"': ["cmd", "v"], "Ctrl-A": ["ctrl", "a"]}
    names |= {"Right arrow": ["right"], "Ctrl + +": ["ctrl", "+"], "PRT SC": ["print_screen"]}
    for text, keys in names.items():
        assert key_tokens(text) == keys and is_key_combination(keys)
    assert not is_key_combination(key_tokens("scholarships for international students"))


def event_types(calls: list[str]) -> list[str]:
    return [event.type for event in parse_code("\n".join(calls))[0]]


def test_videocua_log_becomes_agentnet_steps():
    def entry(kind, timestamp, **parameters):
        return {"action_type": kind, "timestamp": timestamp, "action_params": parameters}

    log = [
        entry("HOTKEY", 1.0, text="Ctrl + O"),
        entry("MOVE_TO", 2.0, x=100, y=50),
        entry("CLICK", 2.1, x=102, y=52, text="Left", numClicks=2),
        entry("MOVE_TO", 3.0, x=200, y=100),
        entry("MOUSE_DOWN", 3.1, text="Left"),
        entry("DRAG_TO", 3.5, x=400, y=300),
        entry("MOUSE_UP", 3.6, text="Left"),
        entry("TYPING", 5.0, text="report"),
        entry("PRESS", 5.5, text="enter"),
    ]
    steps, unsupported = videocua.merge_actions(log, width=1000, height=500)
    assert unsupported == 0 and [step.timestamp for step in steps] == [1.0, 2.0, 3.0, 5.0, None]
    assert [event_types(step.calls) for step in steps] == [
        ["hotkey"],
        ["doubleClick"],
        ["moveTo", "dragTo"],
        ["write", "press"],
        ["terminate"],
    ]
    drag = parse_code("\n".join(steps[2].calls))[0][1]
    assert (drag.x, drag.y, drag.x2, drag.y2) == (0.2, 0.2, 0.4, 0.6)
    assert videocua.with_application("Open the help guide.", "7-Zip") == "In 7-Zip, open the help guide."
    assert videocua.with_application("Open GIMP preferences", "GIMP") == "Open GIMP preferences"
    assert videocua.unwrapped('23,"Right-click an editor tab."') == "Right-click an editor tab."


@pytest.mark.parametrize(
    "code,calls,outcome",
    [
        ("swipe(direction='up', amount=0.7)", ["pyautogui.scroll(-7)"], "continues"),
        ("hotkey('1', '2', '8', '0')", ['pyautogui.write("1280")'], "continues"),
        ("response(answer='No')", [scalecua.TERMINATE], "ended"),
        ("terminate(status='failure')", [], "failed"),
        ("moveTo(x=1440, y=900)", ["pyautogui.moveTo(x=0.5, y=0.5)"], "continues"),
    ],
)
def test_scalecua_code_is_normalized(code, calls, outcome):
    assert scalecua.normalize_code(code, 2880, 1800) == (calls, outcome)


def test_gui360_actions():
    size = (1000, 500)
    click = {"function": "click", "args": {"button": "left", "double": True}, "coordinate_x": 500, "coordinate_y": 100}
    assert gui360.action_calls(click, size) == ["pyautogui.doubleClick(x=0.5, y=0.2)"]
    typed = {"function": "type", "args": {"text": "hello"}, "coordinate_x": 100, "coordinate_y": 100}
    assert event_types(gui360.action_calls(typed, size)) == ["click", "write"]
    assert event_types(gui360.action_calls({"function": "select_table_range", "args": {}}, size)) == ["other"]
    assert gui360.action_calls({"function": "", "args": {}}, size) == [gui360.TERMINATE]


# ------------------------------------------------------------------------------------- rules for the test set
TEST_APPLICATIONS = ["domain:thunderbird", "app:obs studio", "web:google docs", "app:calendar"]


def agentnet_session(session_id: str, split: str, instruction: str, *app_keys: str) -> dict:
    session = {"id": session_id, "split": split, "instruction": instruction, "group": group_hash(instruction)}
    return {**session, "app": app_keys[0], "app_keys": list(app_keys)}


AGENTNET = [
    agentnet_session("train", "train", "Crop the photo", "app:gimp", "web:google", "app:vlc media player"),
    agentnet_session("val", "val", "Blur the background of the photo", "app:gimp"),
    agentnet_session("test1", "test", "Open the settings of Thunderbird", "domain:thunderbird", "web:google"),
    agentnet_session("test2", "test", "Share the planning document", "web:google docs", "app:calendar"),
    # QGIS is held out although it is not a test application: only a test session uses it. VLC is not: training has it.
    agentnet_session("test3", "test", "Record the map window", "app:obs studio", "app:qgis desktop", "web:vlc"),
]


def added_session(instruction="Rename the selected layer to Sky", app="app:gimp", split="train") -> dict:
    return agentnet_session("added", split, instruction, app)


def test_prepared_sets_that_follow_the_rules_for_the_test_set():
    assert not any(violations(AGENTNET, [added_session()], TEST_APPLICATIONS).values())


@pytest.mark.parametrize(
    "agentnet,added,rule",
    [
        ([agentnet_session("x", "test", "Crop a photo", "app:gimp")], [], "every test session has a test application"),
        ([agentnet_session("x", "train", "Record it", "app:obs studio")], [], "no training or validation session"),
        ([agentnet_session("x", "train", "Record the map window", "app:gimp")], [], "no instruction is in two splits"),
        ([], [added_session(split="test")], "every session is training data"),
        ([], [added_session(app="app:mozilla thunderbird")], "no session has a held-out application"),
        ([], [added_session(app="app:qgis")], "no session has a held-out application"),
        ([], [added_session("Write the summary in Google Docs")], "no instruction names a held-out application"),
        ([], [added_session("record the  map window")], "validation or test instruction"),
        ([], [added_session("Record the map window.")], "validation or test instruction"),
    ],
)
def test_sessions_that_break_a_rule_for_the_test_set(agentnet, added, rule):
    broken = {name: ids for name, ids in violations(AGENTNET + agentnet, added, TEST_APPLICATIONS).items() if ids}
    assert [name for name in broken if rule in name] == list(broken) and len(broken) == 1


# ---------------------------------------------------------------------------------------------- cleaning rules
def screenshot(content: int = 0, spread: float = 30.0, size=(1920, 1080)) -> dict:
    return {"size": list(size), "digest": f"{content:032x}", "spread": spread}


def candidate(
    session_id="a",
    instruction="Rename the selected layer to Sky",
    applications=("GIMP",),
    codes=("pyautogui.click(x=0.2, y=0.3)", "pyautogui.write('Sky')", "computer.terminate()"),
    screenshots=None,
) -> clean.Candidate:
    steps, cursor = [], None
    for code in codes:
        events, cursor = parse_code(code, cursor)
        steps.append(
            {"image": "x", "events": [event for event in events if event.type != "terminate"], "focus": cursor}
        )
    screenshots = screenshots if screenshots is not None else [screenshot(index) for index in range(len(codes))]
    return clean.Candidate(session_id, instruction, group_hash(instruction), list(applications), steps, screenshots)


def reason(session: clean.Candidate, kept: clean.Kept | None = None) -> str | None:
    return clean.rejection_reason(session, HeldOut(AGENTNET, TEST_APPLICATIONS), kept or clean.Kept())


@pytest.mark.parametrize(
    "changes",
    [
        dict(),
        dict(applications=("VLC Media Player",)),
        dict(instruction="Add the meeting to the calendar view of the project"),
        dict(instruction="Invite ana@thunderbird.net to the workspace"),
        dict(screenshots=[screenshot(0, spread=1.4), screenshot(1), screenshot(2)]),
        dict(screenshots=[screenshot(index, size=(1028, 1080)) for index in range(3)]),
    ],
)
def test_sessions_that_are_kept(changes):
    assert reason(candidate(**changes)) is None


@pytest.mark.parametrize(
    "rule,changes",
    [
        ("held_out_application", dict(applications=("Mozilla Thunderbird",))),
        ("held_out_application", dict(applications=("QGIS",))),
        ("mentions_held_out_application", dict(instruction="Write the summary in a Google Docs file")),
        ("evaluation_instruction", dict(instruction="record the map  window")),
        ("near_evaluation_instruction", dict(instruction="Record the map window.")),
        ("bad_instruction", dict(instruction="Open")),
        ("missing_screenshot", dict(screenshots=[screenshot(0), None, screenshot(2)])),
        ("too_short", dict(codes=("pyautogui.click(x=0.2, y=0.3)",), screenshots=[screenshot()])),
        ("too_long", dict(codes=("pyautogui.click(x=0.2, y=0.3)",) * 101)),
        ("portrait_screen", dict(screenshots=[screenshot(index, size=(1080, 1920)) for index in range(3)])),
        ("few_input_actions", dict(codes=("computer.api()", "computer.api()", "computer.terminate()"))),
        ("pointer_off_screen", dict(codes=("pyautogui.click(x=5, y=7)", "pyautogui.click(x=3, y=9)", "computer.x()"))),
        ("blank_screenshots", dict(screenshots=[screenshot(0, spread=0.1), screenshot(1), screenshot(2)])),
        ("identical_screenshots", dict(screenshots=[screenshot(7)] * 3)),
    ],
)
def test_sessions_that_are_dropped(rule, changes):
    assert reason(candidate(**changes)) == rule


def test_duplicates_of_kept_sessions_are_dropped():
    kept = clean.Kept()
    kept.add(candidate("a"))
    other_screens = [screenshot(index + 10) for index in range(3)]
    assert reason(candidate("a", screenshots=other_screens, codes=("pyautogui.click()",) * 3), kept) == "duplicate_id"
    assert reason(candidate("b"), kept) == "duplicate_screenshots"
    assert reason(candidate("b", screenshots=other_screens), kept) == "duplicate_demonstration"

    def variant(index: int) -> clean.Candidate:
        codes = ("pyautogui.click(x=0.2, y=0.3)", f"pyautogui.write('Sky {index}')", "computer.terminate()")
        return candidate(f"v{index}", codes=codes, screenshots=[screenshot(20 + 3 * index + step) for step in range(3)])

    for index in range(clean.MAX_SESSIONS_PER_INSTRUCTION - 1):
        assert reason(variant(index), kept) is None
        kept.add(variant(index))
    assert reason(variant(8), kept) == "instruction_repeated_too_often"


def test_every_rule_has_a_test():
    source = Path(__file__).read_text()
    assert all(f'"{rule}"' in source for rules in clean.RULES for rule in rules)


# ----------------------------------------------------------------------------------------------------- loading
@pytest.fixture(scope="module")
def data_config(tmp_path_factory) -> DataConfig:
    root = tmp_path_factory.mktemp("data")
    make_synthetic(root)
    add_synthetic_datasets(root)
    return DataConfig(root=str(root), bulk_root=str(root / "bulk"))


def test_training_data_ids():
    assert Experiment(training_data="all").training_datasets == ("agentnet", "videocua", "scalecua", "gui360")
    assert Experiment.from_id("data-agentnet+gui360").training_datasets == ("agentnet", "gui360")
    assert not Experiment().trains_on_added_data
    for bad in ("videocua", "agentnet+other", "agentnet+"):
        with pytest.raises(ValueError):
            Experiment(training_data=bad)


def test_added_sessions_follow_agentnet_and_only_train(data_config):
    from wfemb.data import TrainingCandidates, WorkflowDataset, evaluation_candidates

    agentnet = WorkflowDataset(data_config, Experiment(**SMALL))
    stacked = WorkflowDataset(data_config, Experiment(training_data="agentnet+gui360", **SMALL))
    n_sessions, n_steps = len(agentnet.sessions), agentnet.frames.n_steps
    for name, array in agentnet.events.items():
        assert np.array_equal(stacked.events[name][: len(array)], array, equal_nan=array.dtype.kind == "f"), name
    assert np.array_equal(stacked.label_keys[:n_sessions], agentnet.label_keys)
    assert np.array_equal(stacked.frames.read(np.arange(40)), agentnet.frames.read(np.arange(40)))
    assert stacked.step_offsets[n_sessions] == n_steps and stacked.frames.n_steps > n_steps
    added_texts = stacked.events["ev_text"][n_steps:]
    assert added_texts[added_texts >= 0].min() >= len(agentnet.text_embeddings)

    for split in ("val", "test"):
        assert np.array_equal(stacked.sessions_in(split), agentnet.sessions_in(split))
    used = np.nonzero(stacked.datasets == "gui360")[0]
    assert len(used) and set(stacked.sessions_in("train")) == set(agentnet.sessions_in("train")) | set(used)
    training = TrainingCandidates(stacked, "cpu")
    assert len(training) == len(agentnet.sessions_in("train")) + len(used)
    assert np.array_equal(evaluation_candidates(stacked, training).session_indices, agentnet.sessions_in("train"))


def test_training_on_added_data(data_config, tmp_path):
    from wfemb.train import training_loop

    config = RunConfig(
        experiment=Experiment(training_data="all", **SMALL),
        train=TrainConfig(max_steps=8, warmup_steps=2),
        eval=EvalConfig(eval_steps=4, n_neighbors=20),
        data=data_config,
        results_root=str(tmp_path),
    )
    final = training_loop(config)
    assert {"val_loss", "test_loss", "test_r@5"} <= set(final)
    assert "data-all" in str(config.run_dir)
