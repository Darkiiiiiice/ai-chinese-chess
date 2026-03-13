.PHONY: install setup play train selfplay evaluate clean help \
        play-visible play-headless train-fast selfplay-fast cycle \
        online-cycle mixed-cycle parity-check

# ============================================================
# 默认值
# ============================================================

# 通用参数
GAMES ?= 1
DEVICE ?= cpu
MODEL ?=

# 在线对弈参数
DIFFICULTY ?= 1
COLOR ?= 1
SIMULATIONS ?= 400
BATCH_SIZE ?= 16
WAIT_TIMEOUT_MS ?= 45000
SPEED_BONUS_MAX ?= 0.3
DRAW_PENALTY ?= -0.5

# 训练参数
EPOCHS ?= 10
TRAIN_BATCH ?= 256
LR ?= 0.001
SAVE_INTERVAL ?= 5

# 自我对弈参数
TEMPERATURE ?= 1.0
MAX_MOVES ?= 300
REPETITION_DRAW_COUNT ?= 6
RESIGN_THRESHOLD ?= -0.95
MIN_RESIGN_MOVES ?= 30
SELFPLAY_WORKERS ?= 1

# 评估参数
EVAL_GAMES ?= 100

# ============================================================
# 安装命令
# ============================================================

install:
	uv sync
	uv run playwright install chromium

setup: install
	mkdir -p ai game browser scripts data models logs

setup-browser:
	uv run playwright install chromium

# ============================================================
# 在线对弈命令
# ============================================================

play-visible:
	uv run python scripts/play.py \
		--model models/model_best.pt \
		--visible \
		--games $(GAMES) \
		--difficulty $(DIFFICULTY) \
		--color $(COLOR) \
		--simulations $(SIMULATIONS) \
		--batch-size $(BATCH_SIZE) \
		--wait-timeout-ms $(WAIT_TIMEOUT_MS) \
		--speed-bonus-max $(SPEED_BONUS_MAX) \
		--draw-penalty $(DRAW_PENALTY) \
		--device $(DEVICE)

play-headless:
	uv run python scripts/play.py \
		--model models/model_best.pt \
		--headless \
		--games $(GAMES) \
		--difficulty $(DIFFICULTY) \
		--color $(COLOR) \
		--simulations $(SIMULATIONS) \
		--batch-size $(BATCH_SIZE) \
		--wait-timeout-ms $(WAIT_TIMEOUT_MS) \
		--speed-bonus-max $(SPEED_BONUS_MAX) \
		--draw-penalty $(DRAW_PENALTY) \
		--device $(DEVICE)

play: play-visible

play-red:
	$(MAKE) play-visible COLOR=1

play-black:
	$(MAKE) play-visible COLOR=-1

parity-check:
	uv run python scripts/check_move_parity.py \
		--games $(GAMES) \
		--plies 40 \
		--difficulty $(DIFFICULTY) \
		--color 1 \
		--headless

# ============================================================
# 训练命令
# ============================================================

train:
	@if [ -n "$(MODEL)" ]; then \
		uv run python scripts/train.py \
			--data data \
			--model $(MODEL) \
			--epochs $(EPOCHS) \
			--batch $(TRAIN_BATCH) \
			--lr $(LR) \
			--save models \
			--interval $(SAVE_INTERVAL) \
			--device $(DEVICE); \
	else \
		uv run python scripts/train.py \
			--data data \
			--epochs $(EPOCHS) \
			--batch $(TRAIN_BATCH) \
			--lr $(LR) \
			--save models \
			--interval $(SAVE_INTERVAL) \
			--device $(DEVICE); \
	fi

train-fast:
	$(MAKE) train EPOCHS=1

# ============================================================
# 自我对弈命令
# ============================================================

selfplay:
	if [ -n "$(MODEL)" ]; then \
		uv run python scripts/self_play.py \
			--model $(MODEL) \
			--games $(GAMES) \
			--simulations $(SIMULATIONS) \
			--temperature $(TEMPERATURE) \
			--max-moves $(MAX_MOVES) \
			--repetition-draw-count $(REPETITION_DRAW_COUNT) \
			--resign-threshold $(RESIGN_THRESHOLD) \
			--min-resign-moves $(MIN_RESIGN_MOVES) \
			--speed-bonus-max $(SPEED_BONUS_MAX) \
			--draw-penalty $(DRAW_PENALTY) \
			--batch-size $(BATCH_SIZE) \
			--num-workers $(SELFPLAY_WORKERS) \
			--device $(DEVICE); \
	else \
		uv run python scripts/self_play.py \
			--games $(GAMES) \
			--simulations $(SIMULATIONS) \
			--temperature $(TEMPERATURE) \
			--max-moves $(MAX_MOVES) \
			--repetition-draw-count $(REPETITION_DRAW_COUNT) \
			--resign-threshold $(RESIGN_THRESHOLD) \
			--min-resign-moves $(MIN_RESIGN_MOVES) \
			--speed-bonus-max $(SPEED_BONUS_MAX) \
			--draw-penalty $(DRAW_PENALTY) \
			--batch-size $(BATCH_SIZE) \
			--num-workers $(SELFPLAY_WORKERS) \
			--device $(DEVICE); \
	fi

selfplay-fast:
	$(MAKE) selfplay GAMES=10 SIMULATIONS=200

# ============================================================
# 评估命令
# ============================================================

evaluate:
	uv run python scripts/evaluate.py \
		--model $(MODEL) \
		--games $(EVAL_GAMES) \
		--simulations $(SIMULATIONS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

# ============================================================
# 训练循环命令
# ============================================================

cycle:
	$(MAKE) selfplay
	$(MAKE) train

online-cycle:
	uv run python scripts/training_loop.py \
		--mode online \
		--model $(MODEL) \
		--online-games $(GAMES) \
		--online-difficulty $(DIFFICULTY) \
		--sp-simulations $(SIMULATIONS) \
		--sp-batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--batch $(TRAIN_BATCH) \
		--lr $(LR) \
		--device $(DEVICE)

mixed-cycle:
	uv run python scripts/training_loop.py \
		--mode mixed \
		--model $(MODEL) \
		--sp-games $(GAMES) \
		--online-games $(GAMES) \
		--online-difficulty $(DIFFICULTY) \
		--sp-simulations $(SIMULATIONS) \
		--sp-batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--batch $(TRAIN_BATCH) \
		--lr $(LR) \
		--device $(DEVICE)

# ============================================================
# 其他命令
# ============================================================

clean:
	rm -rf data/* models/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ============================================================
# 帮助
# ============================================================

help:
	@echo "============================================================"
	@echo "           AlphaZero 中国象棋 - 可用命令"
	@echo "============================================================"
	@echo ""
	@echo "【安装】"
	@echo "  make install        安装依赖 (uv + Playwright)"
	@echo "  make setup          完整设置 (install + 创建目录)"
	@echo "  make setup-browser  仅安装 Playwright 浏览器"
	@echo ""
	@echo "【在线对弈】"
	@echo "  make play           与网页 AI 对弈 (可见模式)"
	@echo "  make play-visible   可见浏览器模式"
	@echo "  make play-headless  无头浏览器模式"
	@echo "  make play-red       执红方 (先手)"
	@echo "  make play-black     执黑方 (后手)"
	@echo "  make parity-check   引擎合法步与网页提示点对拍"
	@echo ""
	@echo "【自我对弈】"
	@echo "  make selfplay       生成自我对弈训练数据"
	@echo "  make selfplay-fast  快速生成 (10 局, 200 模拟)"
	@echo ""
	@echo "【训练】"
	@echo "  make train          训练模型"
	@echo "  make train-fast     快速训练 (1 轮)"
	@echo "  make evaluate       评估模型性能"
	@echo ""
	@echo "【训练循环】"
	@echo "  make cycle          自我对弈 + 训练"
	@echo "  make online-cycle   在线对弈 + 训练"
	@echo "  make mixed-cycle    在线 + 自我对弈 + 训练"
	@echo ""
	@echo "【其他】"
	@echo "  make clean          清理生成的文件"
	@echo "  make help           显示此帮助信息"
	@echo ""
	@echo "============================================================"
	@echo "                      变量参数"
	@echo "============================================================"
	@echo ""
	@echo "【通用参数】"
	@echo "  GAMES=N             对局数"
	@echo "                      默认: 1"
	@echo "                      范围: 1+"
	@echo "                      作用: 控制 selfplay/play 的对局数量"
	@echo ""
	@echo "  DEVICE=cpu|cuda     计算设备"
	@echo "                      默认: cpu"
	@echo "                      范围: cpu, cuda, cuda:0, cuda:1, ..."
	@echo "                      作用: 指定模型推理和训练的设备"
	@echo ""
	@echo "  MODEL=path          模型路径"
	@echo "                      默认: (空)"
	@echo "                      范围: 有效的模型文件路径"
	@echo "                      作用: 加载预训练模型用于 selfplay/train"
	@echo ""
	@echo "【在线对弈参数】"
	@echo "  DIFFICULTY=N        网页 AI 难度等级"
	@echo "                      默认: 1"
	@echo "                      范围: 1-10 (1=最简单, 10=最难)"
	@echo "                      作用: 设置网页对手的 AI 强度"
	@echo ""
	@echo "  COLOR=1|-1          执棋方"
	@echo "                      默认: 1"
	@echo "                      范围: 1=红方(先手), -1=黑方(后手)"
	@echo "                      作用: 指定我方执红或执黑"
	@echo ""
	@echo "  SIMULATIONS=N       MCTS 模拟次数"
	@echo "                      默认: 400"
	@echo "                      范围: 100-2000"
	@echo "                      作用: 每步棋的 MCTS 搜索次数，越多越强但越慢"
	@echo ""
	@echo "  BATCH_SIZE=N        推理批量大小"
	@echo "                      默认: 16"
	@echo "                      范围: 1-128"
	@echo "                      作用: GPU 并行推理批量，GPU 推荐 32-64"
	@echo ""
	@echo "  WAIT_TIMEOUT_MS=N   等待对手超时 (毫秒)"
	@echo "                      默认: 45000"
	@echo "                      范围: 10000-300000"
	@echo "                      作用: 等待网页 AI 落子的最大时间"
	@echo ""
	@echo "  SPEED_BONUS_MAX=N   速胜奖励上限"
	@echo "                      默认: 0.3"
	@echo "                      范围: 0.0-1.0"
	@echo "                      作用: 快速获胜的额外奖励"
	@echo ""
	@echo "  DRAW_PENALTY=N      和棋惩罚"
	@echo "                      默认: -0.5"
	@echo "                      范围: -2.0-0.0"
	@echo "                      作用: 和棋时对双方的惩罚值"
	@echo ""
	@echo "【训练参数】"
	@echo "  EPOCHS=N            训练轮数"
	@echo "                      默认: 10"
	@echo "                      范围: 1-100"
	@echo "                      作用: 每次训练的 epoch 数量"
	@echo ""
	@echo "  TRAIN_BATCH=N       训练批量大小"
	@echo "                      默认: 256"
	@echo "                      范围: 32-2048"
	@echo "                      作用: 训练时的 batch size，GPU 推荐 512-1024"
	@echo ""
	@echo "  LR=N                学习率"
	@echo "                      默认: 0.001"
	@echo "                      范围: 0.00001-0.01"
	@echo "                      作用: 优化器学习率，训练后期建议减小"
	@echo ""
	@echo "  SAVE_INTERVAL=N     模型保存间隔"
	@echo "                      默认: 5"
	@echo "                      范围: 1-100"
	@echo "                      作用: 每隔多少 epoch 保存一次模型"
	@echo ""
	@echo "【自我对弈参数】"
	@echo "  TEMPERATURE=N       探索温度"
	@echo "                      默认: 1.0"
	@echo "                      范围: 0.0-2.0"
	@echo "                      作用: 控制走棋随机性，0=确定性，高=更多探索"
	@echo ""
	@echo "  MAX_MOVES=N         最大步数限制"
	@echo "                      默认: 300"
	@echo "                      范围: 50-500"
	@echo "                      作用: 单局游戏最大半步数"
	@echo ""
	@echo "  REPETITION_DRAW_COUNT=N"
	@echo "                      重复局面判和次数"
	@echo "                      默认: 6"
	@echo "                      范围: 3-10"
	@echo "                      作用: 同一局面出现 N 次后判和"
	@echo ""
	@echo "  RESIGN_THRESHOLD=N  认输阈值"
	@echo "                      默认: -0.95"
	@echo "                      范围: -1.0-0.0"
	@echo "                      作用: 预测胜率低于此值时认输，-1.1 禁用"
	@echo ""
	@echo "  MIN_RESIGN_MOVES=N  最少步数后才能认输"
	@echo "                      默认: 30"
	@echo "                      范围: 0-100"
	@echo "                      作用: 防止过早认输"
	@echo ""
	@echo "【评估参数】"
	@echo "  EVAL_GAMES=N        评估对局数"
	@echo "                      默认: 100"
	@echo "                      范围: 10-1000"
	@echo "                      作用: 评估模型时的对局数量"
	@echo ""
	@echo "============================================================"
	@echo "                      使用示例"
	@echo "============================================================"
	@echo ""
	@echo "  # 在线对弈 (GPU 加速)"
	@echo "  make play GAMES=10 DIFFICULTY=5 DEVICE=cuda BATCH_SIZE=32"
	@echo ""
	@echo "  # 训练 (GPU 加速)"
	@echo "  make train EPOCHS=20 DEVICE=cuda TRAIN_BATCH=512 LR=0.0001"
	@echo ""
	@echo "  # 自我对弈 (使用指定模型)"
	@echo "  make selfplay GAMES=100 MODEL=models/model_best.pt DEVICE=cuda"
	@echo ""
	@echo "  # 完整训练循环"
	@echo "  make cycle GAMES=50 EPOCHS=5 DEVICE=cuda"
	@echo ""
	@echo "  # 在线训练循环"
	@echo "  make online-cycle GAMES=20 DIFFICULTY=5 EPOCHS=5 DEVICE=cuda"
	@echo ""
