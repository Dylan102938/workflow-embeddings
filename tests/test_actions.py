from wfemb.prep.actions import parse_code


def test_click_and_cursor():
    events, cursor = parse_code("pyautogui.click(x=0.1632, y=0.2711)")
    assert [event.type for event in events] == ["click"] and cursor == (0.1632, 0.2711)


def test_drag_uses_pointer_start():
    events, cursor = parse_code("pyautogui.moveTo(x=0.078, y=0.381)\npyautogui.dragTo(x=0.085, y=0.361, button='left')")
    assert [event.type for event in events] == ["moveTo", "dragTo"]
    assert (events[1].x, events[1].y, events[1].x2, events[1].y2) == (0.078, 0.381, 0.085, 0.361)
    assert cursor == (0.085, 0.361)


def test_scroll_hscroll():
    events, _ = parse_code("pyautogui.moveTo(x=0.0745, y=0.5831)\npyautogui.hscroll(13)\npyautogui.scroll(-26)")
    assert [(event.type, event.scroll, event.x) for event in events] == [
        ("moveTo", None, 0.0745),
        ("hscroll", 13.0, 0.0745),
        ("scroll", -26.0, 0.0745),
    ]


def test_keys_text_terminate():
    events, _ = parse_code("pyautogui.hotkey(['ctrl', 'o'])")
    assert events[0].type == "hotkey" and events[0].keys == ["ctrl", "o"]
    events, _ = parse_code("pyautogui.press('enter')")
    assert events[0].keys == ["enter"]
    events, _ = parse_code("pyautogui.write(message='=SUM(F2:F7)')\npyautogui.press('enter')")
    assert events[0].type == "write" and events[0].text == "=SUM(F2:F7)" and events[1].type == "press"
    events, _ = parse_code("computer.terminate(status='failure')")
    assert events[0].type == "terminate"
    events, _ = parse_code("computer.tripleClick(x=0.226, y=0.311)")
    assert events[0].type == "tripleClick"


def test_code_inside_write_is_not_parsed_as_calls():
    events, _ = parse_code("pyautogui.write(message='print(len(x))\\nmain()')")
    assert [event.type for event in events] == ["write"]


def test_click_variants_and_unparseable():
    events, _ = parse_code("pyautogui.click(x=0.5, y=0.5, button='right')")
    assert events[0].type == "rightClick"
    events, _ = parse_code("pyautogui.click(x=0.5, y=0.5, clicks=2)")
    assert events[0].type == "doubleClick"
    events, _ = parse_code("pyautogui.click(x=0.2, y=0.3)\npyautogui.write(message='unterminated")
    assert events[0].type == "click"
