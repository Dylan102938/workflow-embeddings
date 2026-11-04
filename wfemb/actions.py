"""Parse AgentNet pyautogui / computer.* action code into structured events.

Event = (type, x, y, x2, y2, scroll, keys, text); coordinates are normalized to [0, 1].
  click/double/triple/right/middle -> type, x, y
  moveTo                           -> type, x, y
  dragTo                           -> type, x, y (start = current pointer), x2, y2 (end)
  scroll/hscroll                   -> type, scroll amount (+ pointer x, y when known)
  press/hotkey/keyDown/keyUp       -> type, keys (list of key tokens)
  write/typewrite                  -> type, text (typed string)
A step can hold several calls. A pointer position is carried across calls and steps.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

EVENT_TYPES = [
    "pad", "click", "doubleClick", "tripleClick", "rightClick", "middleClick", "moveTo", "dragTo",
    "scroll", "hscroll", "press", "hotkey", "write", "wait", "keyDown", "keyUp", "mouseDown",
    "mouseUp", "terminate", "other",
]
TYPE_ID = {t: i for i, t in enumerate(EVENT_TYPES)}

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


@dataclass
class Event:
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


def _clip01(v) -> float | None:
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return min(max(v, 0.0), 1.0)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _calls_from_ast(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else ""
            name = f.attr
        elif isinstance(f, ast.Name):
            base, name = "", f.id
        else:
            continue
        if base not in ("pyautogui", "computer", ""):
            continue
        if base == "" and name not in TYPE_ID and name not in _ALIASES:
            continue
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
        yield node.lineno, node.col_offset, name, args, kwargs


_CALL_RE = re.compile(r"(?:pyautogui|computer)\.(\w+)\((.*)\)\s*$")


def _calls_from_regex(code: str):
    """Fallback for code that does not parse (rare): one call per line, literal args only."""
    for i, line in enumerate(code.splitlines()):
        m = _CALL_RE.search(line.strip())
        if not m:
            continue
        name, argstr = m.group(1), m.group(2)
        try:
            node = ast.parse(f"f({argstr})", mode="eval").body
            args = [_literal(a) for a in node.args]
            kwargs = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
        except Exception:
            text = re.search(r"message\s*=\s*(['\"])(.*)", argstr)
            args, kwargs = [], ({"message": text.group(2)[:-2]} if text else {})
        yield i, 0, name, args, kwargs


def _get(args, kwargs, pos, *names):
    for n in names:
        if n in kwargs and kwargs[n] is not None:
            return kwargs[n]
    if pos is not None and len(args) > pos:
        return args[pos]
    return None


def _keys(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_keys(x))
        return out
    return [str(v).strip().lower()]


def parse_code(code: str, cursor: tuple[float, float] | None = None):
    """Parse one step's action code. Returns (events, cursor_after)."""
    try:
        calls = sorted(_calls_from_ast(code), key=lambda c: (c[0], c[1]))
    except SyntaxError:
        calls = list(_calls_from_regex(code))
    events: list[Event] = []
    for _, _, name, args, kwargs in calls:
        name = _ALIASES.get(name, name)
        x = _clip01(_get(args, kwargs, 0, "x"))
        y = _clip01(_get(args, kwargs, 1, "y"))
        if name in ("click", "doubleClick", "tripleClick", "rightClick", "middleClick"):
            t = name
            if name == "click":
                button = str(kwargs.get("button") or "left").lower()
                clicks = kwargs.get("clicks") or 1
                if button == "right":
                    t = "rightClick"
                elif button == "middle":
                    t = "middleClick"
                elif clicks == 2:
                    t = "doubleClick"
                elif isinstance(clicks, int) and clicks >= 3:
                    t = "tripleClick"
            if x is None or y is None:
                x, y = cursor if cursor else (None, None)
            events.append(Event(t, x=x, y=y))
            if x is not None:
                cursor = (x, y)
        elif name == "moveTo":
            events.append(Event("moveTo", x=x, y=y))
            if x is not None and y is not None:
                cursor = (x, y)
        elif name == "dragTo":
            sx, sy = cursor if cursor else (None, None)
            events.append(Event("dragTo", x=sx, y=sy, x2=x, y2=y))
            if x is not None and y is not None:
                cursor = (x, y)
        elif name in ("scroll", "hscroll"):
            amt = _get(args, kwargs, 0, "clicks", "amount")
            sx = _clip01(kwargs.get("x"))
            sy = _clip01(kwargs.get("y"))
            if sx is None or sy is None:
                sx, sy = cursor if cursor else (None, None)
            try:
                amt = float(amt)
            except (TypeError, ValueError):
                amt = 0.0
            events.append(Event(name, x=sx, y=sy, scroll=amt))
        elif name in ("press", "keyDown", "keyUp"):
            keys = _keys(_get(args, kwargs, 0, "keys", "key"))
            presses = kwargs.get("presses")
            if isinstance(presses, int) and presses > 1 and keys:
                keys = keys * min(presses, 4)
            events.append(Event(name, keys=keys))
        elif name == "hotkey":
            keys = _keys(list(args)) if args else _keys(kwargs.get("keys"))
            events.append(Event("hotkey", keys=keys))
        elif name == "write":
            text = _get(args, kwargs, 0, "message", "text")
            events.append(Event("write", text=None if text is None else str(text)))
        elif name in ("mouseDown", "mouseUp"):
            if x is None or y is None:
                x, y = cursor if cursor else (None, None)
            events.append(Event(name, x=x, y=y))
        elif name == "wait":
            events.append(Event("wait"))
        elif name == "terminate":
            events.append(Event("terminate"))
        else:
            events.append(Event("other"))
    return events, cursor
