"""Monte Carlo Tree Search for AlphaZero with batch inference support"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from game.engine import GameState
from ai.model import AlphaZero


class MCTSNode:
    """MCTS Node with virtual loss support for parallel search"""

    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Tuple[int, int, int, int]] = None,
        prior: float = 0.0,
        copy_state: bool = True,
    ):
        self.state = state.copy() if copy_state else state
        self.parent = parent
        self.move = move

        self.children: Dict[Tuple[int, int, int, int], MCTSNode] = {}
        self._valid_moves: Optional[List[Tuple[int, int, int, int]]] = None

        self.Q = 0.0  # Mean value
        self.N = 0  # Visit count
        self.P = prior  # Prior probability

        self.virtual_loss = 0  # Virtual loss for parallel search

    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """Return cached valid moves for this node's state."""
        if self._valid_moves is None:
            self._valid_moves = self.state.get_all_valid_moves()
        return self._valid_moves

    def is_expanded(self) -> bool:
        """Check if node is fully expanded"""
        valid_moves = self.get_valid_moves()
        return len(self.children) >= len(valid_moves)

    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0

    def get_Q(self) -> float:
        """Get mean Q value with virtual loss"""
        if self.N + self.virtual_loss == 0:
            return 0.0
        # Virtual loss makes the node less attractive during parallel search
        return (self.Q * self.N - self.virtual_loss) / (self.N + self.virtual_loss)

    def get_UCB(self, c_puct: float, parent_N: int) -> float:
        """Get UCB value"""
        return self.get_Q() + c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)

    def select_child(self, c_puct: float = 1.41) -> Optional["MCTSNode"]:
        """Select child with highest UCB value"""
        best_child = None
        best_ucb = -float("inf")

        for child in self.children.values():
            ucb = child.get_UCB(c_puct, self.N)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def expand(self, moves: List[Tuple[int, int, int, int]], priors: np.ndarray):
        """Expand node with children"""
        for move, prior in zip(moves, priors):
            if move not in self.children:
                new_state = self.state.copy()
                if new_state.do_move(move):
                    self.children[move] = MCTSNode(
                        new_state,
                        self,
                        move,
                        prior,
                        copy_state=False,
                    )

    def backup(self, value: float):
        """Back up value to root"""
        node = self
        while node is not None:
            node.N += 1
            node.Q += (value - node.Q) / node.N  # Running average
            node.virtual_loss = max(0, node.virtual_loss - 1)
            node = node.parent

    def add_virtual_loss(self):
        """Add virtual loss to discourage other threads"""
        self.virtual_loss += 1


class MCTS:
    """Monte Carlo Tree Search with batch inference support"""

    def __init__(
        self,
        model: AlphaZero,
        num_simulations: int = 800,
        c_puct: float = 1.41,
        dirichlet_alpha: float = 0.03,
        epsilon: float = 0.25,
        temperature: float = 1.0,
        device: str = "cpu",
        batch_size: int = 16,  # Batch size for parallel inference
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size

        # Move encoding
        self._init_move_encoder()

    def _init_move_encoder(self):
        """Initialize move encoder"""
        self.move_to_idx = {}
        self.idx_to_move = {}

        idx = 0
        for x1 in range(9):
            for y1 in range(10):
                for x2 in range(9):
                    for y2 in range(10):
                        if (x1, y1) != (x2, y2):
                            move = (x1, y1, x2, y2)
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        self.num_moves = idx

    def encode_move(self, move: Tuple[int, int, int, int]) -> int:
        """Encode move to index"""
        return self.move_to_idx.get(move, 0)

    def decode_move(self, idx: int) -> Tuple[int, int, int, int]:
        """Decode index to move"""
        return self.idx_to_move.get(idx, (0, 0, 0, 0))

    def _visits_to_policy(self, visits: np.ndarray, temperature: float) -> np.ndarray:
        """Convert MCTS visit counts to a policy distribution."""
        if temperature == 0:
            policy = np.zeros(self.num_moves)
            if visits.sum() > 0:
                best_idx = np.argmax(visits)
                policy[best_idx] = 1.0
            return policy

        visits = visits ** (1.0 / temperature)
        if visits.sum() > 0:
            return visits / visits.sum()
        return np.ones(self.num_moves) / self.num_moves

    def _run_search(self, state: GameState) -> np.ndarray:
        """Run one MCTS search and return root visit counts."""
        root = MCTSNode(state)

        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size
        simulations_done = 0

        for _ in range(num_batches):
            batch_count = min(self.batch_size, self.num_simulations - simulations_done)
            self._run_batch_simulation(root, batch_count)
            simulations_done += batch_count

        visits = np.zeros(self.num_moves)
        for move, child in root.children.items():
            idx = self.encode_move(move)
            visits[idx] = child.N

        return visits

    def get_policy(self, state: GameState, temperature: float = None) -> np.ndarray:
        """
        Get action probabilities from MCTS using batch inference

        Args:
            state: Current game state
            temperature: Temperature for softmax (0 = argmax)

        Returns:
            Policy probabilities for all moves
        """
        if temperature is None:
            temperature = self.temperature
        visits = self._run_search(state)
        return self._visits_to_policy(visits, temperature)

    def get_move_and_policy(
        self,
        state: GameState,
        temperature: float = None,
        policy_temperature: float = 0.0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """Run one MCTS search and return both sampled move and record policy."""
        if temperature is None:
            temperature = self.temperature

        visits = self._run_search(state)
        move_policy = self._visits_to_policy(visits.copy(), temperature)
        record_policy = self._visits_to_policy(visits.copy(), policy_temperature)

        move_idx = np.random.choice(self.num_moves, p=move_policy)
        move = self.decode_move(move_idx)

        valid_moves = state.get_all_valid_moves()
        if move not in valid_moves:
            for m in valid_moves:
                idx = self.encode_move(m)
                if move_policy[idx] > 0:
                    move = m
                    break
            else:
                move = valid_moves[0] if valid_moves else None

        return move, record_policy

    def _run_batch_simulation(self, root: MCTSNode, batch_size: int):
        """Run multiple simulations in parallel using batch inference"""
        # Collect leaf nodes and their paths
        leaf_nodes: List[MCTSNode] = []
        paths: List[List[MCTSNode]] = []

        for _ in range(batch_size):
            leaf, path = self._select_leaf(root)
            if leaf is not None:
                leaf_nodes.append(leaf)
                paths.append(path)

        if not leaf_nodes:
            return

        # Batch evaluate all leaf nodes
        boards = []
        valid_moves_list = []

        for leaf in leaf_nodes:
            board = leaf.state.to_numpy()
            boards.append(board)
            valid_moves_list.append(leaf.get_valid_moves())

        # Stack boards into batch
        boards_batch = np.stack(boards, axis=0)

        # Batch inference
        policies, values = self.model.predict_batch(boards_batch)

        # Expand and backup all nodes
        for i, (leaf, path, valid_moves) in enumerate(
            zip(leaf_nodes, paths, valid_moves_list)
        ):
            self._expand_and_backup(leaf, path, valid_moves, policies[i], values[i])

    def _select_leaf(self, root: MCTSNode) -> Tuple[Optional[MCTSNode], List[MCTSNode]]:
        """Select a leaf node for evaluation"""
        node = root
        path = [node]

        # Selection - traverse tree until we find a leaf
        while not node.is_leaf() and node.is_expanded():
            child = node.select_child(self.c_puct)
            if child is None:
                break
            child.add_virtual_loss()
            node = child
            path.append(node)

        # Check if game is over
        if node.state.is_game_over():
            value = node.state.get_game_result()
            if value is None:
                value = 0
            # Backup immediately for terminal nodes
            for n in reversed(path):
                n.virtual_loss = max(0, n.virtual_loss - 1)
            node.backup(value if value is not None else 0)
            return None, []

        # Check if no valid moves
        valid_moves = node.get_valid_moves()
        if len(valid_moves) == 0:
            value = node.state.get_game_result()
            if value is None:
                value = 0
            for n in reversed(path):
                n.virtual_loss = max(0, n.virtual_loss - 1)
            node.backup(value)
            return None, []

        return node, path

    def _expand_and_backup(
        self,
        leaf: MCTSNode,
        path: List[MCTSNode],
        valid_moves: List[Tuple[int, int, int, int]],
        policy: np.ndarray,
        value: float,
    ):
        """Expand leaf node and backup value"""
        # Get priors for valid moves
        move_indices = [self.encode_move(m) for m in valid_moves]
        move_priors = np.array([policy[idx] for idx in move_indices])

        # Normalize
        if move_priors.sum() > 0:
            move_priors = move_priors / move_priors.sum()
        else:
            move_priors = np.ones(len(valid_moves)) / len(valid_moves)

        # Apply Dirichlet noise at root for self-play exploration.
        if leaf.parent is None and len(valid_moves) > 0 and self.epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
            move_priors = (1 - self.epsilon) * move_priors + self.epsilon * noise
            move_priors = move_priors / move_priors.sum()

        # Expand
        leaf.expand(valid_moves, move_priors)

        # Backup
        leaf.backup(value)

    def get_best_move(
        self, state: GameState, temperature: float = None
    ) -> Tuple[int, int, int, int]:
        """Get best move from MCTS"""
        policy = self.get_policy(state, temperature)

        # Sample from policy
        move_idx = np.random.choice(self.num_moves, p=policy)
        move = self.decode_move(move_idx)

        # Validate move
        valid_moves = state.get_all_valid_moves()
        if move not in valid_moves:
            for m in valid_moves:
                idx = self.encode_move(m)
                if policy[idx] > 0:
                    return m
            return valid_moves[0] if valid_moves else None

        return move


class MCTSPlayer:
    """MCTS Player wrapper"""

    def __init__(
        self,
        model: AlphaZero,
        num_simulations: int = 800,
        c_puct: float = 1.41,
        dirichlet_alpha: float = 0.03,
        epsilon: float = 0.25,
        temperature: float = 1.0,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size

        self.mcts = None
        self._init_mcts()

    def _init_mcts(self):
        """Initialize MCTS"""
        self.mcts = MCTS(
            self.model,
            self.num_simulations,
            self.c_puct,
            self.dirichlet_alpha,
            self.epsilon,
            self.temperature,
            self.device,
            self.batch_size,
        )

    def get_move(
        self, state: GameState, temperature: float = None
    ) -> Tuple[int, int, int, int]:
        """Get best move for current state"""
        return self.mcts.get_best_move(state, temperature)

    def get_policy(self, state: GameState, temperature: float = None) -> np.ndarray:
        """Get MCTS policy for current state"""
        return self.mcts.get_policy(state, temperature)

    def get_move_and_policy(
        self,
        state: GameState,
        temperature: float = None,
        policy_temperature: float = 0.0,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """Get move and policy from one MCTS search."""
        return self.mcts.get_move_and_policy(state, temperature, policy_temperature)

    def reset(self):
        """Reset MCTS"""
        self._init_mcts()
