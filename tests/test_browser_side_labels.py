from browser.automate import XiangqiBrowser


def test_describe_sides_for_red():
    mine, opponent = XiangqiBrowser.describe_sides(1)
    assert mine == "红方"
    assert opponent == "黑方"


def test_describe_sides_for_black():
    mine, opponent = XiangqiBrowser.describe_sides(-1)
    assert mine == "黑方"
    assert opponent == "红方"


def test_is_our_turn_for_red_side():
    browser = XiangqiBrowser(player_color=1)
    assert browser.is_our_turn(1) is True
    assert browser.is_our_turn(-1) is False


def test_is_our_turn_for_black_side():
    browser = XiangqiBrowser(player_color=-1)
    assert browser.is_our_turn(1) is False
    assert browser.is_our_turn(-1) is True
