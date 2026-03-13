import numpy as np

from ai.mcts import MCTS, MCTSNode
from game.engine import GameState


class _FakeModel:
    def predict_batch(self, boards):
        batch = boards.shape[0]
        return np.zeros((batch, 8010), dtype=np.float32), np.zeros((batch,), dtype=np.float32)


def test_root_expansion_applies_dirichlet_noise(monkeypatch):
    def _fake_dirichlet(alpha):
        return np.array([0.8, 0.2], dtype=np.float32)

    monkeypatch.setattr(np.random, "dirichlet", _fake_dirichlet)

    mcts = MCTS(
        model=_FakeModel(),
        num_simulations=1,
        dirichlet_alpha=0.03,
        epsilon=1.0,
    )

    root = MCTSNode(GameState())
    moves = root.state.get_all_valid_moves()[:2]
    policy = np.zeros(mcts.num_moves, dtype=np.float32)

    mcts._expand_and_backup(root, [root], moves, policy, 0.0)

    assert np.isclose(root.children[moves[0]].P, 0.8, atol=1e-6)
    assert np.isclose(root.children[moves[1]].P, 0.2, atol=1e-6)
