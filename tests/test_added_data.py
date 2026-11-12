"""Converters, cleaning rules and loading of the datasets added to AgentNet."""

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
from wfemb.prep.sources import gui360, scalecua, videocua  # noqa: E402
from wfemb.prep.sources.common import is_key_combination, key_tokens  # noqa: E402

SMALL = dict(d_model=64, d_head=16, n_heads=4, n_layers=2, n_patches=16, batch_size=16, n_epochs=1, learning_rate=1e-3)


# -------------------------------------------------------------------------------------------------- converters
@pytest.mark.parametrize(
    "text,keys",
    [
        ("Ctrl + Shift+I", ["ctrl", "shift", "i"]),
        ('"Cmd" + "V"', ["cmd", "v"]),
        ("Ctrl-A", ["ctrl", "a"]),
        ("Right arrow", ["right"]),
        ("ctrl + comma", ["ctrl", ","]),
        ("Ctrl + +", ["ctrl", "+"]),
        ("PRT SC", ["print_screen"]),
        ("ENTER", ["enter"]),
    ],
)
def test_key_names_follow_agentnet(text, keys):
    assert key_tokens(text) == keys and is_key_combination(keys)


def test_free_text_in_a_key_field_is_not_a_key():
    assert not is_key_combination(key_tokens("scholarships for international students"))


def action(kind, timestamp, **parameters):
    return {"action_type": kind, "timestamp": timestamp, "action_params": parameters}


def test_videocua_log_becomes_agentnet_steps():
    log = [
        action("HOTKEY", 1.0, text="Ctrl + O"),
        action("MOVE_TO", 2.0, x=100, y=50),
        action("CLICK", 2.1, x=102, y=52, text="Left", numClicks=2),
        action("MOVE_TO", 3.0, x=200, y=100),
        action("MOUSE_DOWN", 3.1, text="Left"),
        action("DRAG_TO", 3.5, x=400, y=300),
        action("MOUSE_UP", 3.6, text="Left"),
        action("TYPING", 5.0, text="report"),
        action("PRESS", 5.5, text="enter"),
    ]
    steps, unsupported = videocua.merge_actions(log, width=1000, height=500)
    assert unsupported == 0
    assert [step.timestamp for step in steps] == [1.0, 2.0, 3.0, 5.0, None]
    event_types = [[event.type for event in parse_code("\n".join(step.calls))[0]] for step in steps]
    assert event_types == [["hotkey"], ["doubleClick"], ["moveTo", "dragTo"], ["write", "press"], ["terminate"]]
    drag = parse_code("\n".join(steps[2].calls))[0][1]
    assert (drag.x, drag.y, drag.x2, drag.y2) == (0.2, 0.2, 0.4, 0.6)


def test_videocua_instruction_names_its_application():
    assert videocua.with_application("Open the help guide.", "7-Zip") == "In 7-Zip, open the help guide."
    assert videocua.with_application("PDF export of the file", "Calibre") == "In Calibre, PDF export of the file"
    assert videocua.with_application("Open GIMP preferences", "GIMP") == "Open GIMP preferences"


@pytest.mark.parametrize(
    "code,calls,outcome",
    [
        ("swipe(direction='up', amount=0.7)", ["pyautogui.scroll(-7)"], "continues"),
        ("hotkey('1', '2', '8', '0')", ['pyautogui.write("1280")'], "continues"),
        ("hotkey('ctrl', 'a')", ['pyautogui.hotkey("ctrl", "a")'], "continues"),
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
    assert [e.type for e in parse_code("\n".join(gui360.action_calls(typed, size)))[0]] == ["click", "write"]
    api = {"function": "select_table_range", "args": {"start_row": 1}}
    assert [e.type for e in parse_code("\n".join(gui360.action_calls(api, size)))[0]] == ["other"]
    assert gui360.action_calls({"function": "", "args": {}}, size) == [gui360.TERMINATE]


# ---------------------------------------------------------------------------------------------------- cleaning
def screenshot(content: int = 0, spread: float = 30.0, size=(1920, 1080)) -> dict:
    return {"size": list(size), "digest": f"{content:032x}", "spread": spread}


def candidate(
    session_id="a",
    instruction="Rename the selected layer to Sky",
    applications=("GIMP",),
    codes=("pyautogui.click(x=0.2, y=0.3)", "pyautogui.write('Sky')", "computer.terminate()"),
    stats=None,
    similarity=0.0,
) -> clean.Candidate:
    steps, cursor = [], None
    for code in codes:
        events, cursor = parse_code(code, cursor)
        steps.append(
            {"image": "x", "events": [event for event in events if event.type != "terminate"], "focus": cursor}
        )
    stats = stats if stats is not None else [screenshot(index) for index in range(len(codes))]
    return clean.Candidate(session_id, instruction, instruction.lower(), list(applications), steps, stats, similarity)


@pytest.fixture
def held_out() -> clean.EvaluationSets:
    sessions = [{"group": "open the settings of thunderbird", "split": "test"}, {"group": "x", "split": "train"}]
    return clean.evaluation_sets(sessions, ["domain:thunderbird", "app:obs studio", "web:google docs", "app:calendar"])


def test_a_good_session_is_kept(held_out):
    assert clean.rejection_reason(candidate(), held_out, clean.Deduplicator()) is None


@pytest.mark.parametrize(
    "changes,reason",
    [
        (dict(instruction="Open"), "bad_instruction"),
        (dict(applications=("Mozilla Thunderbird",)), "held_out_application"),
        (dict(applications=("OBS Studio",)), "held_out_application"),
        (dict(instruction="Write the summary in a Google Docs file"), "mentions_held_out_application"),
        (dict(instruction="Open the settings of Thunderbird"), "mentions_held_out_application"),
        (dict(similarity=0.95), "near_evaluation_instruction"),
        (dict(stats=[screenshot(0), None, screenshot(2)]), "missing_screenshot"),
        (dict(codes=("pyautogui.click(x=0.2, y=0.3)",), stats=[screenshot()]), "too_short"),
        (dict(stats=[screenshot(i, size=(1080, 1920)) for i in range(3)]), "portrait_screen"),
        (dict(codes=("computer.api()", "computer.api()", "computer.terminate()")), "few_input_actions"),
        (
            dict(codes=("pyautogui.click(x=5, y=7)", "pyautogui.click(x=3, y=9)", "computer.terminate()")),
            "pointer_off_screen",
        ),
        (dict(stats=[screenshot(0, spread=0.1), screenshot(1), screenshot(2)]), "blank_screenshots"),
        (dict(stats=[screenshot(7)] * 3), "identical_screenshots"),
    ],
)
def test_rejections(held_out, changes, reason):
    assert clean.rejection_reason(candidate(**changes), held_out, clean.Deduplicator()) == reason


@pytest.mark.parametrize(
    "instruction",
    ["Add the meeting to the calendar view of the project", "Invite ana@thunderbird.net to the workspace"],
)
def test_ordinary_words_and_mail_addresses_are_not_mentions(held_out, instruction):
    assert clean.rejection_reason(candidate(instruction=instruction), held_out, clean.Deduplicator()) is None


def test_a_dark_screen_and_a_tall_window_are_kept(held_out):
    dark_theme = [screenshot(0, spread=1.4), screenshot(1), screenshot(2)]
    assert clean.rejection_reason(candidate(stats=dark_theme), held_out, clean.Deduplicator()) is None
    tall_window = [screenshot(index, size=(1028, 1080)) for index in range(3)]
    assert clean.rejection_reason(candidate(stats=tall_window), held_out, clean.Deduplicator()) is None


def test_duplicates_and_repeated_instructions(held_out):
    seen = clean.Deduplicator()
    seen.accept(candidate("a"))
    assert clean.rejection_reason(candidate("a"), held_out, seen) == "duplicate_id"
    assert clean.rejection_reason(candidate("b"), held_out, seen) == "duplicate_screenshots"
    other_screens = [screenshot(index + 10) for index in range(3)]
    assert clean.rejection_reason(candidate("b", stats=other_screens), held_out, seen) == "duplicate_demonstration"

    def variant(index: int) -> clean.Candidate:
        codes = ("pyautogui.click(x=0.2, y=0.3)", f"pyautogui.write('Sky {index}')", "computer.terminate()")
        return candidate(f"v{index}", codes=codes, stats=[screenshot(20 + 3 * index + step) for step in range(3)])

    for index in range(clean.MAX_SESSIONS_PER_INSTRUCTION - 1):
        assert clean.rejection_reason(variant(index), held_out, seen) is None
        seen.accept(variant(index))
    assert clean.rejection_reason(variant(8), held_out, seen) == "instruction_repeated_too_often"


@pytest.mark.parametrize(
    "name,canonical",
    [
        ("Mozilla Thunderbird", "thunderbird"),
        ("domain:thunderbird", "thunderbird"),
        ("app:microsoft windows", "windows"),
    ]
    + [
        ("IntelliJ IDEA Community Edition", "intellij idea"),
        ("VLC media player", "vlc"),
        ("libreoffice_calc", "libreoffice calc"),
    ],
)
def test_canonical_application_names(name, canonical):
    assert clean.canonical_application(name) == canonical


def test_similarity_to_evaluation_instructions():
    evaluation = ["Open the file report.docx in LibreOffice Writer and make the title bold"]
    instructions = [
        "open the file report.docx in libreoffice writer and make the title bold.",
        "Crop the image to a square",
    ]
    similarities = clean.similarity_to_evaluation(instructions, evaluation)
    assert similarities[0] > clean.EVALUATION_SIMILARITY > similarities[1]


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


def test_added_sessions_follow_agentnet(data_config):
    from wfemb.data import TrainingCandidates, WorkflowDataset, evaluation_candidates

    agentnet = WorkflowDataset(data_config, Experiment(**SMALL))
    stacked = WorkflowDataset(data_config, Experiment(training_data="agentnet+gui360", **SMALL))
    n_sessions, n_steps = len(agentnet.sessions), agentnet.frames.n_steps
    assert len(stacked.sessions) > n_sessions and stacked.frames.n_steps > n_steps

    for name, array in agentnet.events.items():
        assert np.array_equal(stacked.events[name][: len(array)], array, equal_nan=array.dtype.kind == "f"), name
    assert np.array_equal(stacked.label_keys[:n_sessions], agentnet.label_keys)
    assert np.array_equal(stacked.frames.read(np.arange(40)), agentnet.frames.read(np.arange(40)))
    for split in ("val", "test"):
        assert np.array_equal(stacked.sessions_in(split), agentnet.sessions_in(split))

    added = np.arange(n_sessions, len(stacked.sessions))
    used = added[stacked.datasets[added] == "gui360"]
    assert set(stacked.sessions_in("train")) == set(agentnet.sessions_in("train")) | set(used)
    assert stacked.step_offsets[n_sessions] == n_steps
    added_texts = stacked.events["ev_text"][n_steps:]
    assert added_texts[added_texts >= 0].min() >= len(agentnet.text_embeddings)
    first_added_steps = np.arange(n_steps, n_steps + 5)
    assert stacked.frames.read(first_added_steps).shape[0] == 5

    training = TrainingCandidates(stacked, "cpu")
    evaluation = evaluation_candidates(stacked, training)
    assert len(training) == len(agentnet.sessions_in("train")) + len(used)
    assert np.array_equal(evaluation.session_indices, agentnet.sessions_in("train"))
    assert evaluation_candidates(agentnet, TrainingCandidates(agentnet, "cpu")).session_indices is not None


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
