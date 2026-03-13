import asyncio

from browser.automate import XiangqiBrowser


class _FakeMouse:
    async def click(self, x: int, y: int):
        return None


class _FakePage:
    def __init__(self):
        self.mouse = _FakeMouse()


class _NoHintBrowser(XiangqiBrowser):
    def __init__(self):
        super().__init__(player_color=1)
        self.page = _FakePage()
        self.game_data = []
        self.current_move_idx = 0

    async def read_board(self):
        return {(0, 0): "r"}

    async def get_piece_screen_position(self, x: int, y: int):
        if (x, y) == (0, 0):
            return (100, 100)
        return None

    async def get_valid_move_hints(self):
        return []

    async def get_click_position(self, x: int, y: int):
        return (100, 200)


def test_execute_move_returns_false_when_no_valid_hints():
    browser = _NoHintBrowser()

    ok = asyncio.run(browser.execute_move(0, 0, 0, 1))

    assert ok is False
    assert browser.game_data == []
