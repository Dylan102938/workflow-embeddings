from wfemb.actions import parse_code


def test_click_and_cursor():
    ev, cur = parse_code("pyautogui.click(x=0.1632, y=0.2711)")
    assert [e.type for e in ev] == ["click"] and cur == (0.1632, 0.2711)


def test_drag_uses_pointer_start():
    ev, cur = parse_code("pyautogui.moveTo(x=0.078, y=0.381)\npyautogui.dragTo(x=0.085, y=0.361, button='left')")
    assert [e.type for e in ev] == ["moveTo", "dragTo"]
    assert (ev[1].x, ev[1].y, ev[1].x2, ev[1].y2) == (0.078, 0.381, 0.085, 0.361)
    assert cur == (0.085, 0.361)


def test_scroll_hscroll():
    ev, _ = parse_code("pyautogui.moveTo(x=0.0745, y=0.5831)\npyautogui.hscroll(13)\npyautogui.scroll(-26)")
    assert [(e.type, e.scroll, e.x) for e in ev] == [("moveTo", None, 0.0745), ("hscroll", 13.0, 0.0745),
                                                      ("scroll", -26.0, 0.0745)]


def test_keys_text_terminate():
    ev, _ = parse_code("pyautogui.hotkey(['ctrl', 'o'])")
    assert ev[0].type == "hotkey" and ev[0].keys == ["ctrl", "o"]
    ev, _ = parse_code("pyautogui.press('enter')")
    assert ev[0].keys == ["enter"]
    ev, _ = parse_code("pyautogui.write(message='=SUM(F2:F7)')\npyautogui.press('enter')")
    assert ev[0].type == "write" and ev[0].text == "=SUM(F2:F7)" and ev[1].type == "press"
    ev, _ = parse_code("computer.terminate(status='failure')")
    assert ev[0].type == "terminate"
    ev, _ = parse_code("computer.tripleClick(x=0.226, y=0.311)")
    assert ev[0].type == "tripleClick"


def test_code_inside_write_is_not_parsed_as_calls():
    ev, _ = parse_code("pyautogui.write(message='print(len(x))\\nmain()')")
    assert [e.type for e in ev] == ["write"]


def test_click_variants_and_unparseable():
    ev, _ = parse_code("pyautogui.click(x=0.5, y=0.5, button='right')")
    assert ev[0].type == "rightClick"
    ev, _ = parse_code("pyautogui.click(x=0.5, y=0.5, clicks=2)")
    assert ev[0].type == "doubleClick"
    ev, _ = parse_code("pyautogui.click(x=0.2, y=0.3)\npyautogui.write(message='unterminated")
    assert ev[0].type == "click"
