"""Online move-parity checker: engine legal moves vs browser move hints."""

import argparse
import asyncio
import random
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser.automate import XiangqiBrowser
from game.engine import GameState


Move = Tuple[int, int, int, int]


def _choose_move(
    engine_moves: Set[Move],
    browser_moves: Set[Move],
) -> Optional[Move]:
    # Prefer intersection to keep playout aligned with website-legal moves.
    candidates = sorted(engine_moves & browser_moves)
    if not candidates:
        candidates = sorted(engine_moves)
    if not candidates:
        return None
    return random.choice(candidates)


async def run_move_parity_check(
    games: int = 1,
    plies: int = 40,
    difficulty: int = 1,
    player_color: int = 1,
    headless: bool = True,
    max_pieces: Optional[int] = None,
    timeout_ms: int = 30000,
) -> dict:
    summary = {
        "games": games,
        "checked_turns": 0,
        "mismatch_turns": 0,
        "timeout_events": 0,
    }

    for game_idx in range(games):
        print(f"\n=== 对拍第 {game_idx + 1}/{games} 局 ===")
        browser = XiangqiBrowser(
            headless=headless,
            model=None,
            player_color=player_color,
            difficulty=difficulty,
            timeout=timeout_ms,
        )

        try:
            await browser.initialize()
            await browser.navigate_to_game()
            await browser.setup_game(difficulty=difficulty, player_color=player_color)

            board_state = await browser.read_board()
            game_state = GameState([["" for _ in range(9)] for _ in range(10)])
            XiangqiBrowser.sync_game_state_from_board(game_state, board_state)

            for ply in range(1, plies + 1):
                if await browser.is_game_over():
                    print(f"  第 {ply} 手前检测到对局结束")
                    break

                if not browser.is_our_turn(game_state.current_player):
                    baseline = XiangqiBrowser.game_state_to_board_dict(game_state)
                    detected = await browser.wait_for_opponent_move(
                        timeout=timeout_ms,
                        baseline_board=baseline,
                    )
                    if not detected:
                        summary["timeout_events"] += 1
                        continue

                    board_state = await browser.read_board()
                    XiangqiBrowser.sync_after_opponent_move(
                        game_state, board_state, browser.player_color
                    )
                    continue

                # Our turn: compare legal move sets.
                board_state = await browser.read_board()
                XiangqiBrowser.sync_game_state_from_board(game_state, board_state)

                engine_moves = set(game_state.get_all_valid_moves())
                browser_moves = await browser.collect_legal_moves_from_hints(
                    board_state=board_state,
                    color=browser.player_color,
                    max_pieces=max_pieces,
                )
                diff = XiangqiBrowser.diff_move_sets(engine_moves, browser_moves)
                missing = len(diff["missing_in_browser"])
                extra = len(diff["extra_in_browser"])
                summary["checked_turns"] += 1

                if missing or extra:
                    summary["mismatch_turns"] += 1
                    print(
                        f"  [第{ply}手] 差异: missing={missing}, extra={extra} "
                        f"(engine={len(engine_moves)}, browser={len(browser_moves)})"
                    )
                else:
                    print(
                        f"  [第{ply}手] 一致: {len(engine_moves)} legal moves"
                    )

                move = _choose_move(engine_moves, browser_moves)
                if move is None:
                    print(f"  [第{ply}手] 无可用落子")
                    break

                x1, y1, x2, y2 = move
                ok = await browser.execute_move(x1, y1, x2, y2)
                if not ok:
                    print(f"  [第{ply}手] 执行失败: {move}")
                    continue

                if not game_state.do_move(move):
                    game_state.current_player = -browser.player_color

            print("  本局对拍完成")
        finally:
            await browser.close()

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Check move parity between engine and browser hints")
    parser.add_argument("--games", type=int, default=1, help="Number of games to check")
    parser.add_argument("--plies", type=int, default=40, help="Max plies to inspect per game")
    parser.add_argument("--difficulty", type=int, default=1, help="Website AI difficulty")
    parser.add_argument("--color", type=int, default=1, choices=[1, -1], help="Our side color")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--visible", action="store_false", dest="headless", help="Run with browser UI")
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=None,
        help="Limit sampled source pieces per compared turn (default: all)",
    )
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Wait timeout in ms")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any mismatch turn is found",
    )

    args = parser.parse_args()

    summary = await run_move_parity_check(
        games=args.games,
        plies=args.plies,
        difficulty=args.difficulty,
        player_color=args.color,
        headless=args.headless,
        max_pieces=args.max_pieces,
        timeout_ms=args.timeout_ms,
    )

    print("\n=== 对拍汇总 ===")
    print(f"检查局数: {summary['games']}")
    print(f"已对拍回合: {summary['checked_turns']}")
    print(f"差异回合: {summary['mismatch_turns']}")
    print(f"等待超时事件: {summary['timeout_events']}")

    if args.strict and summary["mismatch_turns"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
