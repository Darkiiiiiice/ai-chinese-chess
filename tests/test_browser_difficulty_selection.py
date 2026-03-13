import asyncio

from browser.automate import XiangqiBrowser


class _RecordingSetupBrowser(XiangqiBrowser):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def _select_difficulty(self):
        self.calls.append(("difficulty", self.difficulty))

    async def _select_color(self, color: str):
        self.calls.append(("color", color))

    async def _click_play(self):
        self.calls.append(("play", None))

    async def _wait_for_engine(self):
        self.calls.append(("wait", None))

    async def _cache_board_box(self):
        self.calls.append(("cache", None))

    async def _resolve_random_player_color(self):
        self.calls.append(("resolve_random", None))
        self.player_color = 1


class _FakeDifficultyItem:
    def __init__(self, index: int, clicks: list[int]):
        self.index = index
        self.clicks = clicks

    async def click(self, timeout: int = 0):
        self.clicks.append(self.index)


class _FakeDifficultyLocator:
    def __init__(self, count: int, clicks: list[int]):
        self._count = count
        self._clicks = clicks

    async def count(self):
        return self._count

    def nth(self, index: int):
        return _FakeDifficultyItem(index, self._clicks)


class _FakeDifficultyPage:
    def __init__(self, item_count: int):
        self.clicks = []
        self.item_count = item_count

    def locator(self, selector: str):
        assert selector == ".all-bots > li .bot-item"
        return _FakeDifficultyLocator(self.item_count, self.clicks)


def test_setup_game_selects_difficulty_before_play():
    browser = _RecordingSetupBrowser()

    asyncio.run(browser.setup_game(difficulty=10, player_color=1))

    assert browser.calls == [
        ("difficulty", 10),
        ("color", "Red"),
        ("play", None),
        ("wait", None),
        ("cache", None),
    ]


def test_setup_game_selects_random_color_and_resolves_side():
    browser = _RecordingSetupBrowser()

    asyncio.run(browser.setup_game(difficulty=10, player_color=0))

    assert browser.calls == [
        ("difficulty", 10),
        ("color", "Random"),
        ("play", None),
        ("wait", None),
        ("cache", None),
        ("resolve_random", None),
    ]
    assert browser.player_color == 1


def test_select_difficulty_clicks_matching_bot_card():
    browser = XiangqiBrowser()
    browser.page = _FakeDifficultyPage(item_count=10)
    browser.difficulty = 10

    asyncio.run(browser._select_difficulty())

    assert browser.page.clicks == [9]


class _ResolvingRandomBrowser(XiangqiBrowser):
    def __init__(self):
        super().__init__(player_color=0)
        self.probed_colors = []

    async def read_board(self):
        return {(0, 6): "p", (4, 9): "k", (4, 0): "K"}

    async def collect_legal_moves_from_hints(self, board_state, color, max_pieces=None):
        self.probed_colors.append(color)
        if color == 1:
            return {(0, 6, 0, 5)}
        return set()


def test_resolve_random_player_color_sets_red_when_red_moves_are_available():
    browser = _ResolvingRandomBrowser()

    asyncio.run(browser._resolve_random_player_color())

    assert browser.probed_colors == [1]
    assert browser.player_color == 1
