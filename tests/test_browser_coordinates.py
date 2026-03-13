import asyncio

from browser.automate import XiangqiBrowser


class _FakeBoard:
    async def bounding_box(self):
        return {"x": 100.0, "y": 50.0, "width": 900.0, "height": 1000.0}


class _FakePage:
    async def query_selector(self, selector: str):
        if selector == "#game-grid":
            return _FakeBoard()
        return None


def test_get_click_position_y_axis_is_top_to_bottom():
    browser = XiangqiBrowser()
    browser.page = _FakePage()

    _, y_top = asyncio.run(browser.get_click_position(0, 0))
    _, y_bottom = asyncio.run(browser.get_click_position(0, 9))

    # Engine coordinates define y=0 at the top (black side), y=9 at the bottom (red side).
    assert y_top < y_bottom


def test_get_click_position_x_axis_is_left_to_right():
    browser = XiangqiBrowser()
    browser.page = _FakePage()

    x_left, _ = asyncio.run(browser.get_click_position(0, 0))
    x_right, _ = asyncio.run(browser.get_click_position(8, 0))

    assert x_left < x_right
