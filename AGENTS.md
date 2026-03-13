# Repository Guidelines

## Project Structure & Module Organization
Core code is split by responsibility:
- `game/`: Xiangqi rules, board state, move legality (`engine.py`, `pieces.py`).
- `ai/`: model and search (`model.py`, `mcts.py`).
- `browser/`: Playwright automation for play.xiangqi.com.
- `scripts/`: runnable entry points (`play.py`, `self_play.py`, `train.py`, `evaluate.py`, `training_loop.py`).
- `data/`, `models/`, `logs/`: generated artifacts (training samples, checkpoints, runtime logs).

Keep new logic inside the matching module; keep `scripts/` as orchestration-only wrappers.

## Build, Test, and Development Commands
Use `make` targets (backed by `uv`) as the default workflow:
- `make install`: sync dependencies and install Chromium for Playwright.
- `make setup-browser`: install browser only.
- `make play-visible GAMES=10 DIFFICULTY=5`: run browser gameplay in headed mode.
- `make selfplay GAMES=100 SIMULATIONS=400`: generate self-play training data.
- `make train EPOCHS=10 DEVICE=cuda`: train from `data/`, save checkpoints to `models/`.
- `make cycle GAMES=50 EPOCHS=5`: self-play followed by training.
- `make evaluate`: run model evaluation script.

## Coding Style & Naming Conventions
Target Python `>=3.10`.
- Formatting: Black (`line-length = 100`).
- Linting: Ruff (`line-length = 100`, `py310`).
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, lowercase module names.
- Keep side effects out of import time; put CLI execution under `if __name__ == "__main__":`.

Suggested checks:
- `uv run ruff check .`
- `uv run black .`

## Testing Guidelines
Pytest is available in dev dependencies.
- Place tests under `tests/` with names like `test_engine.py`.
- Name test functions `test_<behavior>()`.
- Prefer deterministic unit tests for `game/` and `ai/`; mock browser interactions where possible.
- Run tests with `uv run pytest -q`.

## Commit & Pull Request Guidelines
No enforced commit format is configured in tooling; use concise Conventional Commit style:
- `feat(ai): improve mcts rollout policy`
- `fix(browser): handle delayed board render`

For PRs, include:
- What changed and why.
- Repro/validation commands run (e.g., `uv run pytest -q`, `make play-headless GAMES=1`).
- Linked issue/task and screenshots for browser-automation UI changes.
