from browser.automate import XiangqiBrowser


def test_diff_move_sets_reports_missing_and_extra():
    engine = {(0, 0, 0, 1), (1, 1, 1, 2), (2, 2, 2, 3)}
    browser = {(0, 0, 0, 1), (2, 2, 2, 3), (5, 5, 5, 6)}

    diff = XiangqiBrowser.diff_move_sets(engine, browser)

    assert diff["missing_in_browser"] == [(1, 1, 1, 2)]
    assert diff["extra_in_browser"] == [(5, 5, 5, 6)]


def test_diff_move_sets_no_difference():
    moves = {(4, 4, 4, 5), (4, 4, 5, 4)}
    diff = XiangqiBrowser.diff_move_sets(moves, set(moves))

    assert diff["missing_in_browser"] == []
    assert diff["extra_in_browser"] == []
