# Xiangqi AlphaZero

AlphaZero-based Chinese Chess (象棋) AI training project.

## 目录

- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [命令详解](#命令详解)
- [参数详解](#参数详解)
- [训练策略](#训练策略)
- [数据格式](#数据格式)

## 项目结构

```
xiangqi_alphazero/
├── ai/                     # 神经网络和MCTS
│   ├── model.py           # AlphaZero 模型 (策略+价值网络)
│   └── mcts.py            # 蒙特卡洛树搜索
├── game/                   # 游戏规则引擎
│   ├── engine.py          # 棋盘状态和规则
│   └── pieces.py          # 棋子定义
├── browser/               # 浏览器自动化
│   └── automate.py        # Playwright 自动化
├── scripts/               # 入口脚本
│   ├── play.py            # 在线对弈
│   ├── train.py           # 训练
│   ├── self_play.py       # 自我对弈
│   ├── evaluate.py        # 评估
│   └── training_loop.py   # 训练循环
├── data/                  # 训练数据
├── models/                # 模型文件
├── logs/                  # 日志
├── Makefile               # 构建脚本
└── README.md              # 本文件
```

## 安装

```bash
# 安装依赖
make install

# 或手动安装
uv sync
uv run playwright install chromium
```

## 快速开始

```bash
# 1. 设置项目
make setup

# 2. 在线对弈 (与网页 AI 对战)
make play GAMES=5 DIFFICULTY=3

# 3. 训练模型
make train EPOCHS=10

# 4. 完整训练循环
make cycle GAMES=100 EPOCHS=10 DEVICE=cuda
```

## 命令详解

### 安装命令

| 命令 | 说明 |
|------|------|
| `make install` | 安装依赖 (uv + Playwright) |
| `make setup` | 完整设置 (install + 创建目录) |
| `make setup-browser` | 仅安装 Playwright 浏览器 |

### 在线对弈命令

| 命令 | 说明 |
|------|------|
| `make play` | 与网页 AI 对弈 (可见模式，默认) |
| `make play-visible` | 可见浏览器模式 |
| `make play-headless` | 无头浏览器模式 (后台运行) |
| `make play-red` | 执红方 (先手) |
| `make play-black` | 执黑方 (后手) |
| `make parity-check` | 引擎合法步与网页提示点对拍 |

### 自我对弈命令

| 命令 | 说明 |
|------|------|
| `make selfplay` | 生成自我对弈训练数据 |
| `make selfplay-fast` | 快速生成数据 (10 局, 200 模拟) |

### 训练命令

| 命令 | 说明 |
|------|------|
| `make train` | 训练模型 (默认 10 轮) |
| `make train-fast` | 快速训练 (1 轮) |
| `make evaluate` | 评估模型性能 |

### 训练循环命令

| 命令 | 说明 |
|------|------|
| `make cycle` | 自我对弈 + 训练 |
| `make online-cycle` | 在线对弈 + 训练 |
| `make mixed-cycle` | 在线 + 自我对弈 + 训练 |

### 其他命令

| 命令 | 说明 |
|------|------|
| `make clean` | 清理生成的文件 |
| `make help` | 显示详细帮助信息 |

## 参数详解

### 通用参数

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `GAMES` | `1` | 1+ | 对局数量 |
| `DEVICE` | `cpu` | `cpu`, `cuda`, `cuda:0`... | 计算设备 |
| `MODEL` | (空) | 文件路径 | 预训练模型路径 |

### 在线对弈参数

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `DIFFICULTY` | `1` | 1-10 | 网页 AI 难度等级 |
| `COLOR` | `1` | `1`, `-1` | 执棋方 (1=红方/先手, -1=黑方/后手) |
| `SIMULATIONS` | `400` | 100-2000 | 每步 MCTS 模拟次数 |
| `BATCH_SIZE` | `16` | 1-128 | 推理批量大小 (GPU 推荐 32-64) |
| `WAIT_TIMEOUT_MS` | `45000` | 10000-300000 | 等待网页 AI 落子超时 (毫秒) |
| `SPEED_BONUS_MAX` | `0.3` | 0.0-1.0 | 速胜奖励上限 |
| `DRAW_PENALTY` | `-0.5` | -2.0-0.0 | 和棋惩罚 |

### 训练参数

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `EPOCHS` | `10` | 1-100 | 训练轮数 |
| `TRAIN_BATCH` | `256` | 32-2048 | 训练批量大小 (GPU 推荐 512-1024) |
| `LR` | `0.001` | 0.00001-0.01 | 学习率 |
| `SAVE_INTERVAL` | `5` | 1-100 | 模型保存间隔 (epoch) |

### 自我对弈参数

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `TEMPERATURE` | `1.0` | 0.0-2.0 | 探索温度 (0=确定性, 高=更多探索) |
| `MAX_MOVES` | `300` | 50-500 | 单局最大步数限制 |
| `REPETITION_DRAW_COUNT` | `6` | 3-10 | 重复局面判和次数 |
| `RESIGN_THRESHOLD` | `-0.95` | -1.0-0.0 | 认输阈值 (-1.1 禁用) |
| `MIN_RESIGN_MOVES` | `30` | 0-100 | 最少步数后才能认输 |

### 评估参数

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `EVAL_GAMES` | `100` | 10-1000 | 评估对局数 |

## 训练策略

### 方式1: 纯自我对弈训练

```bash
# 1. 生成自我对弈数据
make selfplay GAMES=100 SIMULATIONS=400 DEVICE=cuda

# 2. 训练模型
make train EPOCHS=10 DEVICE=cuda

# 3. 评估模型
make evaluate
```

### 方式2: 在线对弈训练

```bash
# 1. 与网页 AI 战斗并收集数据
make play GAMES=20 DIFFICULTY=5 DEVICE=cuda BATCH_SIZE=32

# 2. 训练模型
make train EPOCHS=5 DEVICE=cuda
```

### 方式3: 迭代训练 (推荐)

```bash
# 第一轮: 自我对弈建立基础知识
make cycle GAMES=100 EPOCHS=10 DEVICE=cuda

# 第二轮: 在线对弈学习实战经验
make online-cycle GAMES=20 DIFFICULTY=5 EPOCHS=5 DEVICE=cuda

# 后续轮: 混合训练
make mixed-cycle GAMES=30 EPOCHS=5 DEVICE=cuda
```

### GPU 优化建议

```bash
# GPU 对弈 (推荐参数)
make play GAMES=50 DIFFICULTY=8 DEVICE=cuda BATCH_SIZE=64

# GPU 训练 (推荐参数)
make train EPOCHS=20 DEVICE=cuda TRAIN_BATCH=1024 LR=0.0005

# GPU 自我对弈 (推荐参数)
make selfplay GAMES=200 DEVICE=cuda BATCH_SIZE=64 SIMULATIONS=800
```

## 数据格式

训练数据保存在 `data/` 目录下，PyTorch 格式：

| 文件类型 | 文件名格式 | 内容 |
|----------|------------|------|
| 自我对弈 | `selfplay_*.pt` | `boards`, `policies`, `values` |
| 在线对弈 | `online_*.pt` | `boards`, `policies`, `values` |

**数据结构:**
- `boards`: `(N, 15, 10, 9)` - 棋盘状态张量
  - 15 个通道: 14 种棋子 + 1 个玩家通道
- `policies`: `(N, 8010)` - 策略分布
- `values`: `(N,)` - 游戏结果奖励

## 示例

```bash
# 使用 GPU 进行 10 局对弈
make play GAMES=10 DIFFICULTY=5 DEVICE=cuda BATCH_SIZE=32

# 完整训练循环
make cycle GAMES=50 EPOCHS=5 DEVICE=cuda

# 使用指定模型进行自我对弈
make selfplay GAMES=100 MODEL=models/model_best.pt DEVICE=cuda

# 在线训练循环 (中等难度)
make online-cycle GAMES=20 DIFFICULTY=5 EPOCHS=5 DEVICE=cuda

# 混合训练循环
make mixed-cycle GAMES=30 EPOCHS=5 DEVICE=cuda

# 快速测试
make selfplay-fast
make train-fast
```

## License

MIT
