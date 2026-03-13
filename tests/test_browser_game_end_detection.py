import asyncio

from browser.automate import XiangqiBrowser


class _FakeGameEndPopupPage:
    async def evaluate(self, script: str):
        # Emulate a page where game end is only detectable if the script
        # explicitly checks the ReactModal-based end-of-game structure.
        return all(
            marker in script
            for marker in (
                "ReactModal__Content--after-open",
                "game-end-widget",
                "end-text",
            )
        )


class _FakeResultPayloadPage:
    def __init__(self, payload):
        self.payload = payload

    async def evaluate(self, script: str):
        return self.payload


def test_is_game_over_detects_reactmodal_end_popup():
    browser = XiangqiBrowser()
    browser.page = _FakeGameEndPopupPage()

    assert asyncio.run(browser.is_game_over()) is True


def test_get_my_game_outcome_from_popup_payload():
    browser = XiangqiBrowser(player_color=1)
    browser.page = _FakeResultPayloadPage({"result": "unknown", "my_outcome": "loss"})

    assert asyncio.run(browser.get_my_game_outcome()) == "loss"


def test_get_game_result_text_maps_my_outcome_to_color_result():
    browser = XiangqiBrowser(player_color=-1)
    browser.page = _FakeResultPayloadPage({"result": "unknown", "my_outcome": "win"})

    assert asyncio.run(browser.get_game_result_text()) == "black_wins"


def test_get_my_game_outcome_falls_back_to_color_result():
    browser = XiangqiBrowser(player_color=-1)
    browser.page = _FakeResultPayloadPage({"result": "red_wins", "my_outcome": "unknown"})

    assert asyncio.run(browser.get_my_game_outcome()) == "loss"
