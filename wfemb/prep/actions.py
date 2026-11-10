"""Parses AgentNet action code (pyautogui.* / computer.* calls) into structured events."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

EVENT_TYPES = [
    "pad",
    "click",
    "doubleClick",
    "tripleClick",
    "rightClick",
    "middleClick",
    "moveTo",
    "dragTo",
    "scroll",
    "hscroll",
    "press",
    "hotkey",
    "write",
    "wait",
    "keyDown",
    "keyUp",
    "mouseDown",
    "mouseUp",
    "terminate",
    "other",
]
TYPE_ID = {event_type: i for i, event_type in enumerate(EVENT_TYPES)}

_ALIASES = {
    "typewrite": "write",
    "triple_click": "tripleClick",
    "double_click": "doubleClick",
    "right_click": "rightClick",
    "middle_click": "middleClick",
    "move_to": "moveTo",
    "drag_to": "dragTo",
    "dragRel": "dragTo",
    "moveRel": "moveTo",
    "key_down": "keyDown",
    "key_up": "keyUp",
}
_CLICK_TYPES = ("click", "doubleClick", "tripleClick", "rightClick", "middleClick")
_MAX_KEY_REPEATS = 4

Cursor = tuple[float, float] | None
Call = tuple[int, int, str, list, dict]  # line, column, function name, positional values, keyword values


@dataclass
class Event:
    """One input action. Coordinates are normalized to [0, 1]; a drag goes from (x, y) to (x2, y2)."""

    type: str
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    scroll: float | None = None
    keys: list[str] = field(default_factory=list)
    text: str | None = None

    @property
    def point(self) -> tuple[float, float] | None:
        """Where the action lands on screen (end point for drags)."""
        if self.x2 is not None and self.y2 is not None:
            return (self.x2, self.y2)
        if self.x is not None and self.y is not None:
            return (self.x, self.y)
        return None


def parse_code(code: str, cursor: Cursor = None) -> tuple[list[Event], Cursor]:
    """Parses one step's action code into (events, cursor after the step); pass the cursor on to the next step."""
    try:
        calls = sorted(_calls_from_ast(code), key=lambda call: (call[0], call[1]))
    except SyntaxError:
        calls = list(_calls_from_regex(code))
    events: list[Event] = []
    for _, _, name, args, kwargs in calls:
        event, cursor = _event_from_call(_ALIASES.get(name, name), args, kwargs, cursor)
        events.append(event)
    return events, cursor


def _event_from_call(name: str, args: list, kwargs: dict, cursor: Cursor) -> tuple[Event, Cursor]:
    x = _clip_to_unit(_get(args, kwargs, 0, "x"))
    y = _clip_to_unit(_get(args, kwargs, 1, "y"))
    has_target = x is not None and y is not None

    if name in _CLICK_TYPES:
        if not has_target:
            x, y = cursor or (None, None)
        if x is not None:
            cursor = (x, y)
        return Event(_click_type(name, kwargs), x=x, y=y), cursor
    if name == "moveTo":
        return Event("moveTo", x=x, y=y), ((x, y) if has_target else cursor)
    if name == "dragTo":
        start_x, start_y = cursor or (None, None)
        return Event("dragTo", x=start_x, y=start_y, x2=x, y2=y), ((x, y) if has_target else cursor)
    if name in ("scroll", "hscroll"):
        return _scroll_event(name, args, kwargs, cursor), cursor
    if name in ("press", "keyDown", "keyUp"):
        keys = _key_tokens(_get(args, kwargs, 0, "keys", "key"))
        presses = kwargs.get("presses")
        if isinstance(presses, int) and presses > 1 and keys:
            keys = keys * min(presses, _MAX_KEY_REPEATS)
        return Event(name, keys=keys), cursor
    if name == "hotkey":
        return Event("hotkey", keys=_key_tokens(list(args)) if args else _key_tokens(kwargs.get("keys"))), cursor
    if name == "write":
        text = _get(args, kwargs, 0, "message", "text")
        return Event("write", text=None if text is None else str(text)), cursor
    if name in ("mouseDown", "mouseUp"):
        if not has_target:
            x, y = cursor or (None, None)
        return Event(name, x=x, y=y), cursor
    if name in ("wait", "terminate"):
        return Event(name), cursor
    return Event("other"), cursor


def _click_type(name: str, kwargs: dict) -> str:
    """click(button=..., clicks=...) becomes the specific click type."""
    if name != "click":
        return name
    button = str(kwargs.get("button") or "left").lower()
    clicks = kwargs.get("clicks") or 1
    if button == "right":
        return "rightClick"
    if button == "middle":
        return "middleClick"
    if clicks == 2:
        return "doubleClick"
    if isinstance(clicks, int) and clicks >= 3:
        return "tripleClick"
    return "click"


def _scroll_event(name: str, args: list, kwargs: dict, cursor: Cursor) -> Event:
    """The scroll position is the x= / y= keywords, else the pointer (positional values are the amount)."""
    x = _clip_to_unit(kwargs.get("x"))
    y = _clip_to_unit(kwargs.get("y"))
    if x is None or y is None:
        x, y = cursor or (None, None)
    amount = _get(args, kwargs, 0, "clicks", "amount")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 0.0
    return Event(name, x=x, y=y, scroll=amount)


def _calls_from_ast(code: str) -> Iterator[Call]:
    """Every pyautogui.* / computer.* call, and bare calls of known action names, in ast.walk order."""
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            module = function.value.id if isinstance(function.value, ast.Name) else ""
            name = function.attr
        elif isinstance(function, ast.Name):
            module, name = "", function.id
        else:
            continue
        if module not in ("pyautogui", "computer", ""):
            continue
        if module == "" and name not in TYPE_ID and name not in _ALIASES:
            continue
        yield node.lineno, node.col_offset, name, *_literal_arguments(node)


_CALL_PATTERN = re.compile(r"(?:pyautogui|computer)\.(\w+)\((.*)\)\s*$")
_MESSAGE_PATTERN = re.compile(r"message\s*=\s*(['\"])(.*)")


def _calls_from_regex(code: str) -> Iterator[Call]:
    """Fallback for code that does not parse: one call per line, literal arguments only."""
    for line_number, line in enumerate(code.splitlines()):
        match = _CALL_PATTERN.search(line.strip())
        if not match:
            continue
        name, argument_source = match.group(1), match.group(2)
        try:
            args, kwargs = _literal_arguments(ast.parse(f"f({argument_source})", mode="eval").body)
        except Exception:
            message = _MESSAGE_PATTERN.search(argument_source)
            # [:-2] is kept as is: changing it changes the typed texts in vocab.json (and their text_vecs rows)
            args, kwargs = [], ({"message": message.group(2)[:-2]} if message else {})
        yield line_number, 0, name, args, kwargs


def _literal_arguments(call: ast.Call) -> tuple[list, dict]:
    """A call's positional and keyword values; anything that is not a literal becomes None."""
    args = [_literal(argument) for argument in call.args]
    kwargs = {keyword.arg: _literal(keyword.value) for keyword in call.keywords if keyword.arg}
    return args, kwargs


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _get(args: list, kwargs: dict, position: int | None, *names: str) -> Any:
    """The first non-None keyword among names, else the positional value at position."""
    # wfemb/demo/raw.py (pixel_coordinates) swaps this function out to rescale pixel coordinates: keep its name,
    # its (args, kwargs, position, *names) signature, and the x / y lookups in _event_from_call going through it.
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    if position is not None and len(args) > position:
        return args[position]
    return None


def _clip_to_unit(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), 1.0)


def _key_tokens(value: Any) -> list[str]:
    """Flattens a key name or nested lists of key names into lowercase tokens."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [token for item in value for token in _key_tokens(item)]
    return [str(value).strip().lower()]
