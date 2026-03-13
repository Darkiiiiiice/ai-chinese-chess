import math

import torch

import scripts.train as train_script
from ai.model import create_model


def test_train_step_policy_loss_stays_finite_with_probability_targets():
    model = create_model({"device": "cpu"})
    trainer = train_script.Trainer(
        model=model,
        learning_rate=0.001,
        batch_size=2,
        device="cpu",
    )

    boards = torch.zeros((2, 15, 10, 9), dtype=torch.float32)
    policies = torch.zeros((2, 8010), dtype=torch.float32)
    policies[0, 0] = 1.0
    policies[1, 1] = 1.0
    values = torch.tensor([1.0, -1.0], dtype=torch.float32)

    stats = trainer.train_step(boards, policies, values)

    assert math.isfinite(stats["policy_loss"])
    assert math.isfinite(stats["loss"])
