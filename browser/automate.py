"""Browser Automation for Xiangqi - Play.xiangqi.com"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Set proxy if needed
PROXY = os.environ.get('https_proxy') or os.environ.get('http_proxy')

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from game.engine import GameState


class XiangqiBrowser:
    """Browser automation for play.xiangqi.com"""

    BASE_URL = "https://play.xiangqi.com/"

    # Piece type mapping (from class name to piece character)
    # Website uses English class names: king, advisor, elephant, rook, cannon, horse, pawn
    PIECE_MAP = {
        'king': 'k',      # 将/帅 (King/General)
        'advisor': 'a',   # 士 (Advisor)
        'elephant': 'e',  # 象 (Elephant)
        'rook': 'r',      # 车 (Rook/Chariot)
        'cannon': 'c',    # 炮 (Cannon)
        'horse': 'h',     # 马 (Horse)
        'pawn': 'p',      # 兵/卒 (Pawn)
    }

    @staticmethod
    def describe_sides(player_color: int) -> Tuple[str, str]:
        """Return (our_side, opponent_side) labels in Chinese."""
        our_side = '红方' if player_color == 1 else '黑方'
        opponent_side = '黑方' if player_color == 1 else '红方'
        return our_side, opponent_side

    def is_our_turn(self, current_player: int) -> bool:
        """Whether current player in game state is our automation-controlled side."""
        return current_player == self.player_color

    @staticmethod
    def sync_game_state_from_board(
        game_state: GameState, board_state: Dict[Tuple[int, int], str]
    ) -> None:
        """Sync full board matrix in GameState from sparse board dict."""
        for y in range(10):
            for x in range(9):
                game_state.board[y][x] = board_state.get((x, y), "")

    @staticmethod
    def sync_after_opponent_move(
        game_state: GameState,
        board_state: Dict[Tuple[int, int], str],
        our_color: Optional[int] = None,
    ) -> None:
        """Apply latest board and advance turn after opponent has moved."""
        XiangqiBrowser.sync_game_state_from_board(game_state, board_state)
        if our_color in (-1, 1):
            # After opponent move is observed, it must be our turn.
            game_state.current_player = our_color
        else:
            game_state.current_player = -game_state.current_player

    @staticmethod
    def game_state_to_board_dict(game_state: GameState) -> Dict[Tuple[int, int], str]:
        """Convert GameState board matrix into sparse board dict."""
        board: Dict[Tuple[int, int], str] = {}
        for y in range(10):
            for x in range(9):
                piece = game_state.board[y][x]
                if piece:
                    board[(x, y)] = piece
        return board

    @staticmethod
    def diff_move_sets(
        engine_moves: Set[Tuple[int, int, int, int]],
        browser_moves: Set[Tuple[int, int, int, int]],
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Diff two move sets and return sorted discrepancies."""
        missing = sorted(engine_moves - browser_moves)
        extra = sorted(browser_moves - engine_moves)
        return {
            "missing_in_browser": missing,
            "extra_in_browser": extra,
        }

    async def collect_legal_moves_from_hints(
        self,
        board_state: Dict[Tuple[int, int], str],
        color: int,
        max_pieces: Optional[int] = None,
    ) -> Set[Tuple[int, int, int, int]]:
        """
        Collect browser-legal moves by selecting each piece and reading move hints.

        Args:
            board_state: Current board map from read_board().
            color: 1 for red, -1 for black.
            max_pieces: Optional cap for number of pieces to sample.
        """
        moves: Set[Tuple[int, int, int, int]] = set()

        piece_positions = [
            (x, y)
            for (x, y), piece in sorted(board_state.items())
            if piece and GameState.get_piece_color(piece) == color
        ]
        if max_pieces is not None:
            piece_positions = piece_positions[:max_pieces]

        for x, y in piece_positions:
            try:
                src = await self.get_piece_screen_position(x, y)
                if not src:
                    continue

                sx, sy = src
                await self.page.mouse.click(sx, sy)
                await asyncio.sleep(0.12)

                hints = await self.get_valid_move_hints()
                for hint in hints:
                    hx, hy = hint.get("x"), hint.get("y")
                    if isinstance(hx, int) and isinstance(hy, int):
                        if 0 <= hx < 9 and 0 <= hy < 10:
                            moves.add((x, y, hx, hy))

                # Deselect current piece to avoid stale highlight influencing next piece.
                await self.page.mouse.click(sx, sy)
                await asyncio.sleep(0.08)
            except Exception:
                continue

        return moves

    async def detect_our_turn_from_hints(
        self,
        board_state: Optional[Dict[Tuple[int, int], str]] = None,
        max_pieces: int = 6,
    ) -> bool:
        """
        Heuristic fallback to detect if it's our turn:
        if selecting our pieces yields any legal move hints, we can move now.
        """
        current_board = board_state or await self.read_board()
        legal_moves = await self.collect_legal_moves_from_hints(
            board_state=current_board,
            color=self.player_color,
            max_pieces=max_pieces,
        )
        return bool(legal_moves)

    def __init__(
        self,
        headless: bool = False,
        model=None,
        player_color: int = 1,  # Our side color: 1 for red, -1 for black
        difficulty: int = 1,
        num_simulations: int = 400,
        batch_size: int = 16,
        timeout: int = 60000,
        proxy: str = None
    ):
        self.headless = headless
        self.model = model
        self.player_color = player_color
        self.difficulty = difficulty
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.timeout = timeout
        self.proxy = proxy or PROXY

        self.page: Optional[Page] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

        # Game state tracking
        self.board_state: Dict[Tuple[int, int], str] = {}
        self.game_data: List[Dict] = []
        self.current_move_idx = 0

        # Board box cache
        self._board_box = None

    async def initialize(self):
        """Initialize browser"""
        playwright = await async_playwright().start()

        launch_options = {'headless': self.headless}
        if self.proxy:
            launch_options['proxy'] = {'server': self.proxy}

        self.browser = await playwright.chromium.launch(**launch_options)

        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800}
        )
        self.page = await self.context.new_page()

        # Set default timeout
        self.page.set_default_timeout(self.timeout)

        print(f"浏览器初始化完成 (无头模式={self.headless}, 代理={self.proxy})")

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        print("浏览器已关闭")

    async def navigate_to_game(self):
        """Navigate to game page and setup"""
        print(f"正在导航到 {self.BASE_URL}")
        await self.page.goto(self.BASE_URL, wait_until='domcontentloaded')
        await asyncio.sleep(2)

        # Handle popup
        await self._handle_popup()

        # Click AI mode
        await self._click_ai_mode()

        # Wait for game options
        await asyncio.sleep(2)

    async def _handle_popup(self):
        """Handle popup dialog and neutralize blocking overlays."""
        closed = False

        # Common consent/notice button texts seen across locales.
        for text in ("Got it", "Accept", "OK", "Close", "I Agree", "知道了", "关闭", "同意"):
            try:
                btn = self.page.get_by_text(text, exact=False).first
                await btn.click(timeout=1200)
                closed = True
                await asyncio.sleep(0.2)
            except Exception:
                pass

        # Extra fallback for explicit close controls in modal/dialog UIs.
        try:
            close_btn = self.page.locator(
                '[aria-label="Close"], .modal-close, .dialog-close, .close-button'
            ).first
            await close_btn.click(timeout=1000)
            closed = True
            await asyncio.sleep(0.2)
        except Exception:
            pass

        # If overlay still exists, make it non-blocking so clicks can pass through.
        try:
            overlays = self.page.locator('.ReactModal__Overlay.ReactModal__Overlay--after-open')
            if await overlays.count() > 0:
                await self.page.evaluate("""
                    () => {
                        document
                            .querySelectorAll('.ReactModal__Overlay.ReactModal__Overlay--after-open')
                            .forEach((el) => {
                                el.style.pointerEvents = 'none';
                            });
                    }
                """)
                print("已清除阻挡的弹窗遮罩")
        except Exception:
            pass

        if closed:
            print("已关闭弹窗")
        await asyncio.sleep(0.4)

    async def _click_ai_mode(self):
        """Click Play Computer button"""
        play_computer = self.page.get_by_text('Play Computer', exact=False).first
        try:
            await self._handle_popup()
            await play_computer.click(timeout=3000)
            print("已点击 Play Computer")
        except Exception as e:
            print(f"点击 AI 模式失败: {e}")
            # One more attempt: clear blocking overlays and force click.
            await self._handle_popup()
            await play_computer.click(timeout=5000, force=True)
            print("已点击 Play Computer (强制)")

    async def setup_game(
        self,
        difficulty: int = None,
        player_color: int = None,
        red_first: bool = True
    ):
        """Setup game with options"""
        if difficulty is not None:
            self.difficulty = difficulty
        if player_color is not None:
            self.player_color = player_color
        else:
            self.player_color = 1 if red_first else -1

        await asyncio.sleep(2)

        # Select color (AI color)
        if self.player_color == -1:
            # AI is black, player selects black -> AI goes first
            await self._select_color('Black')
        else:
            await self._select_color('Red')

        await asyncio.sleep(1)

        # Click Play to start
        await self._click_play()

        # Wait for engine initialization
        await self._wait_for_engine()

        # Cache board box
        await self._cache_board_box()

    async def _select_color(self, color: str):
        """Select player color"""
        try:
            label = self.page.get_by_text(color, exact=True)
            await label.click(timeout=3000)
            print(f"已选择 {color}")
        except Exception as e:
            print(f"选择颜色失败: {e}")

    async def _click_play(self):
        """Click Play button"""
        try:
            play_btn = self.page.get_by_text('Play', exact=True).last
            await play_btn.click(timeout=3000)
            print("已点击 Play")
        except Exception as e:
            print(f"点击 Play 失败: {e}")

    async def _wait_for_engine(self):
        """Wait for engine to initialize"""
        print("等待引擎初始化...")
        for i in range(60):
            await asyncio.sleep(1)
            try:
                text = await self.page.query_selector('body')
                content = await text.inner_text()
                if 'Initialising' not in content:
                    print(f"引擎就绪，耗时 {i+1} 秒")
                    await asyncio.sleep(3)  # Extra time for pieces to load
                    return
            except:
                pass
        print("警告: 引擎初始化超时")

    async def _cache_board_box(self):
        """Cache board bounding box for coordinate calculations"""
        board = await self.page.query_selector('#game-grid')
        if board:
            self._board_box = await board.bounding_box()

    async def read_board(self) -> Dict[Tuple[int, int], str]:
        """
        Read current board state from browser

        Returns:
            Dict mapping (x, y) to piece char (e.g., 'r', 'R', 'p', 'P')
            - Lowercase = Red pieces
            - Uppercase = Black pieces
        """
        board = {}

        try:
            # Get all pieces from the page
            board_info = await self.page.evaluate('''
                () => {
                    const result = [];
                    const pieces = document.querySelectorAll('.piece');
                    const board = document.getElementById('game-grid');
                    if (!board) return result;

                    const boardRect = board.getBoundingClientRect();
                    const cellWidth = boardRect.width / 9;

                    pieces.forEach((piece) => {
                        // Get the wrapper container
                        const wrapper = piece.closest('.Wrapper__PieceWrapper');
                        if (!wrapper) {
                            // Try alternative container
                            const container = piece.parentElement?.parentElement;
                            if (!container) return;

                            const r = parseInt(container.getAttribute('r') || '0');
                            const rect = container.getBoundingClientRect();
                            const c = Math.floor((rect.x - boardRect.x) / cellWidth) + 1;

                            const img = piece.querySelector('.img-holder');
                            let pieceType = '';
                            if (img) {
                                const match = img.className.match(/(king|advisor|elephant|rook|cannon|horse|pawn)/);
                                if (match) pieceType = match[1];
                            }

                            const cls = piece.className || '';
                            let color = '';
                            if (cls.includes('red-piece')) {
                                color = 'red';
                            } else if (cls.includes('brown-piece') || cls.includes('black-piece')) {
                                color = 'black';
                            }

                            result.push({r: r, c: c, type: pieceType, color: color});
                            return;
                        }

                        // Get position from wrapper
                        const r = parseInt(wrapper.getAttribute('r') || '0');
                        const c = parseInt(wrapper.getAttribute('c') || '0');

                        // Determine piece type and color from class names
                        const cls = piece.className || '';
                        let pieceType = '';
                        let color = '';

                        // Get piece type from img-holder class
                        const img = piece.querySelector('.img-holder');
                        if (img) {
                            const imgClass = img.className || '';
                            const match = imgClass.match(/(king|advisor|elephant|rook|cannon|horse|pawn)/);
                            if (match) pieceType = match[1];
                        }

                        // Determine color
                        if (cls.includes('red-piece')) {
                            color = 'red';
                        } else if (cls.includes('brown-piece') || cls.includes('black-piece')) {
                            color = 'black';
                        }

                        result.push({
                            r: r,
                            c: c,
                            type: pieceType,
                            color: color
                        });
                    });
                    return result;
                }
            ''')

            # Convert to board format
            for info in board_info:
                # r is 1-10 (1=bottom, 10=top)
                # c is 1-9 (1=left, 9=right)
                # Our engine uses: x (0-8, 0=left), y (0-9, 0=black side/bottom)
                x = info['c'] - 1  # 1->0, 9->8
                y = info['r'] - 1  # 1->9, 10->0 (需要翻转)

                # Actually, based on the web structure:
                # r=1 is bottom (red side for initial setup)
                # r=10 is top (black side for initial setup)
                # Our engine: y=0 is black side (top), y=9 is red side (bottom)
                # So we need to flip: y = 10 - r
                y = 10 - info['r']  # r=1 -> y=9, r=10 -> y=0

                piece_type = self.PIECE_MAP.get(info['type'], '')

                if info['color'] == 'red':
                    piece = piece_type.lower()
                elif info['color'] == 'black':
                    piece = piece_type.upper()
                else:
                    piece = ''

                if piece and 0 <= x < 9 and 0 <= y < 10:
                    board[(x, y)] = piece

        except Exception as e:
            print(f"读取棋盘错误: {e}")

        return board

    async def get_click_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert board coordinates to pixel coordinates

        Args:
            x: Board column (0-8, left to right)
            y: Board row (0-9, 0=black side/top, 9=red side/bottom)

        Returns:
            (px, py) pixel coordinates for clicking
        """
        # Get board container
        board = await self.page.query_selector('#game-grid')
        if not board:
            return 0, 0

        box = await board.bounding_box()
        if not box:
            return 0, 0

        # Board dimensions: 9 columns, 10 rows
        cell_width = box['width'] / 9
        cell_height = box['height'] / 10

        # Coordinate conversion:
        # Our engine: y=0 is black side (top), y=9 is red side (bottom).
        # Screen/web y increases downward from top.
        px = box['x'] + (x + 0.5) * cell_width
        py = box['y'] + (y + 0.5) * cell_height

        return int(px), int(py)

    async def get_piece_screen_position(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        Get screen coordinates for a piece at board position (x, y)
        Uses the same method as reference xiangqi/scripts/auto_play_xiangqi.py

        Args:
            x: Board column (0-8, left to right)
            y: Board row (0-9, 0=black side/top, 9=red side/bottom)

        Returns:
            (px, py) screen coordinates or None if no piece found
        """
        pos = await self.page.evaluate('''
            (coords) => {
                const {x, y} = coords;

                // Get all piece images (class contains "-zh" like "rook-red-zh")
                const imgs = document.querySelectorAll('img[class*="-zh"]');
                if (!imgs.length) return null;

                // Calculate board boundaries from actual piece positions
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                imgs.forEach(img => {
                    const rect = img.getBoundingClientRect();
                    minX = Math.min(minX, rect.x + rect.width/2);
                    maxX = Math.max(maxX, rect.x + rect.width/2);
                    minY = Math.min(minY, rect.y + rect.height/2);
                    maxY = Math.max(maxY, rect.y + rect.height/2);
                });

                const colWidth = (maxX - minX) / 8;
                const rowHeight = (maxY - minY) / 9;

                // Calculate expected position for (x, y)
                // Our coords: x=0-8 (left to right), y=0-9 (0=black side/top, 9=red side/bottom)
                // Screen: y increases downward
                // y=0 (top/black) -> minY, y=9 (bottom/red) -> maxY
                const targetX = minX + x * colWidth;
                const targetY = minY + y * rowHeight;

                // Find the piece closest to this position
                let closest = null;
                let minDist = Infinity;

                imgs.forEach(img => {
                    const rect = img.getBoundingClientRect();
                    const imgCenterX = rect.x + rect.width/2;
                    const imgCenterY = rect.y + rect.height/2;

                    const dist = Math.sqrt(
                        Math.pow(imgCenterX - targetX, 2) +
                        Math.pow(imgCenterY - targetY, 2)
                    );

                    if (dist < minDist && dist < colWidth * 0.6) {
                        minDist = dist;
                        closest = {
                            x: Math.round(imgCenterX),
                            y: Math.round(imgCenterY)
                        };
                    }
                });

                return closest;
            }
        ''', {'x': x, 'y': y})

        if pos:
            return (pos['x'], pos['y'])
        return None

    async def click_position(self, x: int, y: int) -> bool:
        """Click on board position using mouse coordinates"""
        pos = await self.get_piece_screen_position(x, y)
        if pos:
            px, py = pos
            print(f"    [点击] 棋盘({x},{y}) -> 屏幕({px},{py}) [有棋子]")
            await self.page.mouse.click(px, py)
            await asyncio.sleep(0.3)
            return True

        # Fallback: click on board cell position (for empty cells or if piece not found)
        px, py = await self.get_click_position(x, y)
        print(f"    [点击] 棋盘({x},{y}) -> 屏幕({px},{py}) [空位/回退]")
        await self.page.mouse.click(px, py)
        await asyncio.sleep(0.3)
        return True

    async def click_empty_position(self, x: int, y: int):
        """Click on an empty board position (for destination)"""
        px, py = await self.get_click_position(x, y)
        print(f"    [点击空位] 棋盘({x},{y}) -> 屏幕({px},{py})")
        await self.page.mouse.click(px, py)
        await asyncio.sleep(0.3)

    async def get_valid_move_hints(self) -> List[Tuple[int, int]]:
        """Get valid move positions after selecting a piece (green dots)"""
        hints = await self.page.evaluate('''
            () => {
                const hints = document.querySelectorAll(
                    '.can-move-to-square, .can-move-to-point, .can-move, [class*="can-move"]'
                );
                if (!hints.length) return [];

                // Get board boundaries from piece images
                const imgs = document.querySelectorAll('img[class*="-zh"]');
                if (!imgs.length) return [];

                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                imgs.forEach(img => {
                    const rect = img.getBoundingClientRect();
                    minX = Math.min(minX, rect.x + rect.width/2);
                    maxX = Math.max(maxX, rect.x + rect.width/2);
                    minY = Math.min(minY, rect.y + rect.height/2);
                    maxY = Math.max(maxY, rect.y + rect.height/2);
                });

                const colWidth = (maxX - minX) / 8;
                const rowHeight = (maxY - minY) / 9;

                const positions = [];
                hints.forEach(hint => {
                    const rect = hint.getBoundingClientRect();
                    const hx = rect.x + rect.width/2;
                    const hy = rect.y + rect.height/2;

                    // Convert to board coordinates
                    const col = Math.round((hx - minX) / colWidth);
                    const row = Math.round((hy - minY) / rowHeight);

                    positions.push({x: col, y: row, screenX: Math.round(hx), screenY: Math.round(hy)});
                });

                return positions;
            }
        ''')
        return hints

    async def execute_move(self, x1: int, y1: int, x2: int, y2: int):
        """Execute a move on the board using mouse clicks"""
        print(f"  执行落子: ({x1},{y1}) -> ({x2},{y2})")
        board_before = await self.read_board()
        moving_piece = board_before.get((x1, y1))

        # Click source piece to select it
        src_pos = await self.get_piece_screen_position(x1, y1)
        if src_pos:
            px, py = src_pos
            print(f"  点击源棋子，屏幕坐标 ({px}, {py})")
            await self.page.mouse.click(px, py)
            await asyncio.sleep(0.5)
        else:
            print(f"  警告: 找不到 ({x1},{y1}) 处的棋子")
            return False

        # Check if piece is selected (look for valid move hints)
        valid_hints = await self.get_valid_move_hints()
        print(f"  找到 {len(valid_hints)} 个可走位置")

        if not valid_hints:
            # Try clicking again
            print(f"  未找到可走位置，尝试重新点击...")
            await self.page.mouse.click(px, py)
            await asyncio.sleep(0.3)
            valid_hints = await self.get_valid_move_hints()
            print(f"  重试后: {len(valid_hints)} 个可走位置")
            if not valid_hints:
                print("  ✗ 未能选中棋子（无可走提示），取消本次落子")
                return False

        # Find the destination in valid hints
        target_hint = None
        for hint in valid_hints:
            if hint['x'] == x2 and hint['y'] == y2:
                target_hint = hint
                break

        if target_hint:
            # Click on the hint (green dot)
            px, py = target_hint['screenX'], target_hint['screenY']
            print(f"  点击可走位置提示 ({px}, {py})")
            await self.page.mouse.click(px, py)
        else:
            # Click destination directly (for capture or if no hint)
            dst_pos = await self.get_piece_screen_position(x2, y2)
            if dst_pos:
                px, py = dst_pos
                print(f"  点击目标位置（吃子），屏幕坐标 ({px}, {py})")
            else:
                px, py = await self.get_click_position(x2, y2)
                print(f"  点击目标位置（空位），屏幕坐标 ({px}, {py})")
            await self.page.mouse.click(px, py)

        # Confirm move is applied on board to avoid false positives.
        moved = False
        for _ in range(6):
            await asyncio.sleep(0.2)
            board_after = await self.read_board()

            if moving_piece:
                src_piece = board_after.get((x1, y1))
                dst_piece = board_after.get((x2, y2))
                if dst_piece == moving_piece and src_piece != moving_piece:
                    moved = True
                    break
            elif board_after != board_before:
                moved = True
                break

        if not moved:
            print("  ✗ 落子未生效，取消本次落子")
            return False

        print(f"  ✓ 落子完成: ({x1},{y1}) -> ({x2},{y2})")

        # Record move
        self.game_data.append({
            'move_idx': self.current_move_idx,
            'from': (x1, y1),
            'to': (x2, y2),
            'player': self.player_color
        })
        self.current_move_idx += 1
        return True

    async def is_my_turn(self) -> bool:
        """Check if it's AI's turn to move"""
        # Check for turn indicator on page
        try:
            # Look for turn indicator
            turn_info = await self.page.evaluate('''
                () => {
                    // Check for turn indicator element
                    const turnIndicator = document.querySelector('.turn-indicator, .current-turn, .player-turn');
                    if (turnIndicator) {
                        const text = turnIndicator.textContent.toLowerCase();
                        if (text.includes('red') || text.includes('红')) return 'red';
                        if (text.includes('black') || text.includes('黑')) return 'black';
                    }

                    // Check for active player indicator
                    const activePlayer = document.querySelector('.active, .current-player, [class*="active"]');
                    if (activePlayer) {
                        const cls = activePlayer.className || '';
                        if (cls.includes('red')) return 'red';
                        if (cls.includes('black') || cls.includes('brown')) return 'black';
                    }

                    // Check body text for turn indicators
                    const body = document.body.innerText;
                    if (body.includes('Your turn') || body.includes('Red to move') || body.includes('红方走')) {
                        return 'red';
                    }
                    if (body.includes('Black to move') || body.includes('黑方走')) {
                        return 'black';
                    }

                    return null;
                }
            ''')

            if turn_info:
                if turn_info == 'red':
                    return self.player_color == 1
                elif turn_info == 'black':
                    return self.player_color == -1

            # Check for last-move highlight (if exists, opponent just moved)
            last_move = await self.page.evaluate('''
                () => {
                    const lastMove = document.querySelector('.last-move, .highlight, [class*="last"]');
                    return lastMove !== null;
                }
            ''')
            # If there's a last-move highlight and it's not our turn indicator,
            # we can't reliably determine, so return None equivalent
        except:
            pass

        # Return None (unknown) - caller should use other detection methods
        return None

    async def wait_for_opponent_move(
        self,
        timeout: int = 30000,
        baseline_board: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> bool:
        """Wait for opponent to make a move"""
        print("等待对手(网页AI)落子...")

        # Store current board state
        old_board = dict(baseline_board) if baseline_board is not None else await self.read_board()
        old_board_str = self._board_to_str(old_board)
        print(f"  当前棋盘状态:\n{old_board_str}")
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout / 1000:
            await asyncio.sleep(0.5)

            # Check for game over
            if await self.is_game_over():
                print("  检测到游戏结束")
                return False

            # Check if it's now our turn (indicates opponent moved)
            my_turn = await self.is_my_turn()
            if my_turn is True:
                print("  轮次指示器显示轮到我们了")
                return True

            # Check if board changed
            new_board = await self.read_board()
            if new_board != old_board:
                # Double-check the change is real (not just read noise)
                await asyncio.sleep(0.3)
                confirm_board = await self.read_board()
                if confirm_board != old_board:
                    print("对手已落子!")
                    print(f"  新棋盘状态:\n{self._board_to_str(confirm_board)}")
                    return True

        # One-shot fallback checks right at timeout boundary.
        try:
            final_board = await self.read_board()
            if final_board != old_board:
                print("  超时边界检测到棋盘变化，判定对手已落子")
                return True

            if await self.detect_our_turn_from_hints(board_state=final_board):
                print("  超时兜底检测到我方可走子，判定轮到我们")
                return True
            if await self.detect_our_turn_from_hints(
                board_state=final_board, max_pieces=None
            ):
                print("  超时兜底全量检测到我方可走子，判定轮到我们")
                return True
        except Exception:
            pass

        print("警告: 等待对手超时")
        return False

    def _board_to_str(self, board: Dict) -> str:
        """Convert board dict to readable string for debugging"""
        lines = []
        for y in range(10):
            row = []
            for x in range(9):
                piece = board.get((x, y), '.')
                row.append(piece if piece else '.')
            lines.append(' '.join(row))
        return '\n'.join(lines)

    @staticmethod
    def _build_game_over_script() -> str:
        return '''
            () => {
                const body = (document.body?.innerText || '').replace(/\\s+/g, ' ');
                const hasAny = (phrases) => phrases.some((phrase) => body.includes(phrase));
                const isVisible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    if (
                        style.display === 'none' ||
                        style.visibility === 'hidden' ||
                        style.opacity === '0'
                    ) {
                        return false;
                    }
                    return el.offsetParent !== null || style.position === 'fixed';
                };

                if (hasAny([
                    'Game Over', 'Checkmate', 'Red wins', 'Black wins', 'Draw',
                    '红方胜', '黑方胜', '和棋', '绝杀', '将死'
                ])) {
                    return true;
                }

                const reactModal = document.querySelector(
                    '.ReactModal__Content.ReactModal__Content--after-open, .ReactModal__Content--after-open'
                );
                const endWidget = document.querySelector('.game-end-widget');
                const endLabel = document.querySelector(
                    '.game-end-widget .end-text, .game-end-widget .end-text.wins, .game-end-widget .end-text.lose'
                );

                if (isVisible(reactModal) && (isVisible(endWidget) || isVisible(endLabel))) {
                    return true;
                }

                if (isVisible(reactModal) && hasAny(['再来一局', '下一级', '复盘', '胜', '负', '对'])) {
                    return true;
                }

                const genericModal = document.querySelector('.game-over, .modal, .dialog');
                if (isVisible(genericModal) && hasAny(['Game Over', '胜', '负', '和棋', 'Draw'])) {
                    return true;
                }

                return false;
            }
        '''

    @staticmethod
    def _build_game_result_script() -> str:
        return '''
            () => {
                const body = (document.body?.innerText || '').replace(/\\s+/g, ' ');
                const hasAny = (phrases) => phrases.some((phrase) => body.includes(phrase));
                const normalize = (value) => (value || '').toString().toLowerCase().replace(/\\s+/g, ' ').trim();
                const isVisible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    if (
                        style.display === 'none' ||
                        style.visibility === 'hidden' ||
                        style.opacity === '0'
                    ) {
                        return false;
                    }
                    return el.offsetParent !== null || style.position === 'fixed';
                };
                const parseOutcomeFromEndText = (el) => {
                    if (!el) return 'unknown';
                    const className = normalize(el.className);
                    const text = normalize(el.textContent);
                    if (className.includes('wins') || text.includes('wins') || text.includes('win') || text.includes('胜')) {
                        return 'win';
                    }
                    if (
                        className.includes('lose') ||
                        text.includes('lose') ||
                        text.includes('loses') ||
                        text.includes('负') ||
                        text.includes('败')
                    ) {
                        return 'loss';
                    }
                    if (text.includes('draw') || text.includes('和棋') || text.includes('平局')) {
                        return 'draw';
                    }
                    return 'unknown';
                };

                if (hasAny(['Red wins', '红方胜', '红方获胜', '红胜'])) {
                    return { result: 'red_wins', my_outcome: 'unknown' };
                }
                if (hasAny(['Black wins', '黑方胜', '黑方获胜', '黑胜'])) {
                    return { result: 'black_wins', my_outcome: 'unknown' };
                }
                if (hasAny(['Draw', '和棋', '平局'])) {
                    return { result: 'draw', my_outcome: 'draw' };
                }

                const reactModal = document.querySelector(
                    '.ReactModal__Content.ReactModal__Content--after-open, .ReactModal__Content--after-open'
                );
                const endWidget = document.querySelector('.game-end-widget');
                const widgets = Array.from(document.querySelectorAll('.game-end-widget .widget'));

                if (isVisible(reactModal) || isVisible(endWidget)) {
                    const meWidget = widgets.find(
                        (widget) =>
                            widget.querySelector('a.profile-link[href]') ||
                            widget.querySelector('h4.username a[href^="/@"]')
                    ) || widgets[0];

                    const myEndText = meWidget ? meWidget.querySelector('.end-text') : null;
                    const myOutcome = parseOutcomeFromEndText(myEndText);
                    return { result: 'unknown', my_outcome: myOutcome };
                }

                return { result: 'unknown', my_outcome: 'unknown' };
            }
        '''

    @staticmethod
    def _normalize_result_payload(payload) -> Dict[str, str]:
        if isinstance(payload, str):
            return {'result': payload, 'my_outcome': 'unknown'}
        if isinstance(payload, dict):
            result = payload.get('result', 'unknown')
            my_outcome = payload.get('my_outcome', 'unknown')
            return {
                'result': result if isinstance(result, str) else 'unknown',
                'my_outcome': my_outcome if isinstance(my_outcome, str) else 'unknown',
            }
        return {'result': 'unknown', 'my_outcome': 'unknown'}

    async def is_game_over(self) -> bool:
        """Check if the game has ended"""
        try:
            game_over = await self.page.evaluate(self._build_game_over_script())
            return game_over
        except:
            return False

    async def get_game_result_text(self) -> str:
        """Get the game result text from the page"""
        try:
            payload = await self.page.evaluate(self._build_game_result_script())
            normalized = self._normalize_result_payload(payload)
            result = normalized['result']
            if result in ('red_wins', 'black_wins', 'draw'):
                return result

            my_outcome = normalized['my_outcome']
            if my_outcome == 'win':
                return 'red_wins' if self.player_color == 1 else 'black_wins'
            if my_outcome == 'loss':
                return 'black_wins' if self.player_color == 1 else 'red_wins'
            if my_outcome == 'draw':
                return 'draw'
            return 'unknown'
        except:
            return 'unknown'

    async def get_my_game_outcome(self) -> str:
        """Get game outcome for our side: win/loss/draw/unknown."""
        try:
            payload = await self.page.evaluate(self._build_game_result_script())
            normalized = self._normalize_result_payload(payload)
            my_outcome = normalized['my_outcome']
            if my_outcome in ('win', 'loss', 'draw'):
                return my_outcome

            # Fallback: infer from color-based result.
            result = normalized['result']
            if result == 'draw':
                return 'draw'
            if result == 'red_wins':
                return 'win' if self.player_color == 1 else 'loss'
            if result == 'black_wins':
                return 'win' if self.player_color == -1 else 'loss'
            return 'unknown'
        except:
            return 'unknown'

    async def play_game(self) -> Dict:
        """Play a complete game"""
        our_side, opponent_side = self.describe_sides(self.player_color)
        print(f"\n开始游戏: 我方执{our_side}，对手执{opponent_side}")

        # Initialize game state
        game_state = GameState()
        self.game_data = []
        self.current_move_idx = 0

        # Read initial board
        self.board_state = await self.read_board()

        # Update game state with board
        self.sync_game_state_from_board(game_state, self.board_state)

        max_moves = 200
        move_count = 0

        while not game_state.is_game_over() and move_count < max_moves:
            move_count += 1

            # Check if game is over in browser
            if await self.is_game_over():
                print("浏览器检测到游戏结束")
                break

            # Check if it's our turn
            current_player = game_state.current_player
            is_our_turn = self.is_our_turn(current_player)

            if is_our_turn:
                print(f"\n我方落子 {move_count} (当前玩家={current_player})")

                # Get move from model
                move = await self._get_ai_move(game_state)

                if move:
                    x1, y1, x2, y2 = move
                    print(f"我方: ({x1},{y1}) -> ({x2},{y2})")

                    # Execute move in browser
                    move_ok = await self.execute_move(x1, y1, x2, y2)
                    if not move_ok:
                        print("浏览器执行落子失败，保持我方回合并重试")
                        self.board_state = await self.read_board()
                        self.sync_game_state_from_board(game_state, self.board_state)
                        game_state.current_player = self.player_color
                        continue

                    # Update local game state
                    if not game_state.do_move(move):
                        # Local engine can be stale vs scraped board; keep turn progression sane.
                        game_state.current_player = -self.player_color
                else:
                    print("未找到有效落子!")
                    break

                # Check game over
                if game_state.is_game_over() or await self.is_game_over():
                    break

                # Wait for opponent
                baseline_board = self.game_state_to_board_dict(game_state)
                detected = await self.wait_for_opponent_move(baseline_board=baseline_board)
                if not detected:
                    continue

                # Update board after opponent move
                self.board_state = await self.read_board()

                # Sync game state with browser and advance turn after opponent move.
                self.sync_after_opponent_move(game_state, self.board_state, self.player_color)
            else:
                # Wait for human move
                print(f"\n等待对手(网页AI)落子 {move_count}")
                baseline_board = self.game_state_to_board_dict(game_state)
                detected = await self.wait_for_opponent_move(baseline_board=baseline_board)
                if not detected:
                    continue

                # Update board after opponent move
                self.board_state = await self.read_board()

                # Sync game state with browser and advance turn after opponent move.
                self.sync_after_opponent_move(game_state, self.board_state, self.player_color)

            await asyncio.sleep(0.5)

        # Get game result
        result = game_state.get_game_result()
        result_text = await self.get_game_result_text()
        my_outcome = await self.get_my_game_outcome()

        # Override result from browser if available
        if result_text == 'red_wins':
            result = 1
        elif result_text == 'black_wins':
            result = -1
        elif result_text == 'draw':
            result = 0

        if result == 1:
            print("\n红方胜利!")
        elif result == -1:
            print("\n黑方胜利!")
        else:
            print("\n和棋!")

        if my_outcome == 'win':
            print("我方结果: 胜")
        elif my_outcome == 'loss':
            print("我方结果: 负")
        elif my_outcome == 'draw':
            print("我方结果: 和")

        # Save game data
        await self._save_game_data(result)

        return {
            'result': result,
            'moves': self.game_data,
            'num_moves': move_count
        }

    async def _get_ai_move(self, game_state: GameState) -> Optional[Tuple[int, int, int, int]]:
        """Get AI move from model"""
        if self.model is None:
            # Random move for testing
            import random
            moves = game_state.get_all_valid_moves()
            if moves:
                return random.choice(moves)
            return None

        # Use MCTS
        from ai.mcts import MCTSPlayer

        mcts = MCTSPlayer(
            model=self.model,
            num_simulations=self.num_simulations,
            temperature=0.0
        )

        move = mcts.get_move(game_state)
        return move

    async def _save_game_data(self, result: int):
        """Save game data to file"""
        import os
        os.makedirs('data', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/game_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump({
                'result': result,
                'player_color': self.player_color,
                'moves': self.game_data,
                'timestamp': timestamp
            }, f, indent=2)

        print(f"游戏数据已保存到 {filepath}")

    async def restart_game(self):
        """Restart the game"""
        await self.navigate_to_game()
        await self.setup_game(player_color=self.player_color)


async def create_browser_automation(
    model=None,
    player_color: int = 1,
    difficulty: int = 1,
    num_simulations: int = 400,
    headless: bool = False
) -> 'XiangqiBrowser':
    """Create and initialize browser automation"""
    browser = XiangqiBrowser(
        headless=headless,
        model=model,
        player_color=player_color,
        difficulty=difficulty,
        num_simulations=num_simulations
    )

    await browser.initialize()
    await browser.navigate_to_game()
    await browser.setup_game(player_color=player_color)

    return browser


async def test_read_board():
    """Test board reading"""
    browser = XiangqiBrowser(headless=False)

    try:
        await browser.initialize()
        await browser.navigate_to_game()
        await browser.setup_game(player_color=-1)  # Black (AI goes first)

        # Read board
        board = await browser.read_board()
        print("\n=== 棋盘状态 ===")
        for y in range(9, -1, -1):
            row = ''
            for x in range(9):
                piece = board.get((x, y), '.')
                row += piece + ' '
            print(f"{y}: {row}")

    finally:
        await browser.close()


async def test_click():
    """Test clicking"""
    browser = XiangqiBrowser(headless=False)

    try:
        await browser.initialize()
        await browser.navigate_to_game()
        await browser.setup_game(player_color=-1)

        # Read board
        board = await browser.read_board()

        # Find a red piece (at bottom)
        for (x, y), piece in board.items():
            if piece.islower():  # Red piece
                print(f"点击红方棋子 ({x}, {y})")
                await browser.click_position(x, y)
                await asyncio.sleep(2)
                await browser.page.screenshot(path='clicked.png')
                print("截图已保存")
                break

    finally:
        await browser.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test_click':
        asyncio.run(test_click())
    else:
        asyncio.run(test_read_board())
