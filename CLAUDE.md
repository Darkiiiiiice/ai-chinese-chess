# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-based Chinese Chess (象棋/Xiangqi) AI training project using:
- Playwright browser automation to play against web AI at play.xiangqi.com
- MCTS (Monte Carlo Tree Search) with neural network guidance
- Self-play and online-play for training data generation

## Commands

```bash
# Setup
make install          # Install dependencies with uv
make setup-browser    # Install Playwright chromium

# Gameplay (online against web AI)
make play             # Visible mode, AI plays red (default)
make play-red         # AI plays red (moves first)
make play-black       # AI plays black (waits for opponent)
make play-visible GAMES=10 DIFFICULTY=5
make play-headless GAMES=10

# Training
make train EPOCHS=10 DEVICE=cuda
make train-fast       # Quick 1-epoch training

# Self-play data generation
make selfplay GAMES=100 SIMULATIONS=400
make selfplay-fast    # Quick 10 games

# Training cycles
make cycle GAMES=50 EPOCHS=5              # Self-play + train
make online-cycle GAMES=10 EPOCHS=5       # Online play + train
make mixed-cycle GAMES=10 EPOCHS=5        # Mixed mode

# Evaluation
make evaluate GAMES=100 DEVICE=cuda BATCH_SIZE=32

# Testing & Linting
uv run pytest -q
uv run ruff check .
uv run black .

# Cleanup
make clean             # Remove data/, models/, logs/
```

## Architecture

### Core Components

- **game/engine.py**: `GameState` - board state, move generation, validation. Board is 9x10 grid (x=0-8, y=0-9 where y=0 is black side). Pieces: lowercase=red, uppercase=black.

- **ai/model.py**: `AlphaZero` wrapper with separate `PolicyNet` and `ValueNet`. Policy output: 8010 moves. Value output: scalar [-1, 1].

- **ai/mcts.py**: `MCTS` core algorithm with UCB selection, virtual loss for parallel search, and batch inference support. `MCTSPlayer` wrapper for getting moves/policies.

- **browser/automate.py**: `XiangqiBrowser` - Playwright automation. `read_board()` parses DOM, `execute_move()` clicks positions.

- **scripts/**: Entry points (`play.py`, `train.py`, `self_play.py`, `evaluate.py`, `training_loop.py`). Keep as orchestration wrappers only.

### Data Flow

```
Self-play / Online-play → data/*.pt (boards, policies, values) → Training → models/*.pt → Inference
```

**Important**: `train.py` only loads `selfplay_*.pt` files (line 32: `pattern = os.path.join(data_dir, "selfplay_*.pt")`). Online play data (`online_*.pt`) is NOT loaded by default.

### Key Details

**Move Encoding**: Index = iterate x1(0-8), y1(0-9), x2(0-8), y2(0-9), skip (x1,y1)==(x2,y2). Total: 8010.

**Board Coordinates**: x=0-8 (left-right), y=0-9 (top-bottom, y=0 is black side). Web r attribute (1-10) maps to y = 10 - r.

**Color Parameter**: `--color 1` means our AI plays red (moves first), `--color -1` means AI plays black (waits for opponent).

### GPU Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | 16 | MCTS batch inference size (GPU: 32-64) |
| `TRAIN_BATCH` | 256 | Training batch size (GPU: 512-1024) |
| `DEVICE` | cpu | Device (cpu/cuda) |

Higher batch sizes improve GPU utilization during MCTS inference and training.

## Style

- Python >=3.10, Black/Ruff with line-length 100
- `snake_case` functions/variables, `PascalCase` classes
- Tests in `tests/` named `test_*.py`
- Keep side effects out of import time; CLI under `if __name__ == "__main__":`
