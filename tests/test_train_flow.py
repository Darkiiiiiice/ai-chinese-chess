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


def test_dataset_loads_online_and_selfplay_files(tmp_path):
    sample = {
        "boards": torch.zeros((1, 15, 10, 9), dtype=torch.float32),
        "policies": torch.zeros((1, 8010), dtype=torch.float32),
        "values": torch.tensor([0.5], dtype=torch.float32),
    }
    torch.save(sample, tmp_path / "selfplay_test.pt")
    torch.save(sample, tmp_path / "online_test.pt")

    dataset = train_script.AlphaZeroDataset(str(tmp_path))

    assert len(dataset) == 2


def test_split_indices_by_source_keeps_sources_disjoint():
    samples = [
        {"source_id": "a"},
        {"source_id": "a"},
        {"source_id": "b"},
        {"source_id": "b"},
        {"source_id": "c"},
    ]

    train_indices, val_indices, grouped = train_script.split_indices_by_source(
        samples,
        val_split=0.3,
        seed=42,
    )

    train_sources = {samples[i]["source_id"] for i in train_indices}
    val_sources = {samples[i]["source_id"] for i in val_indices}

    assert grouped is True
    assert train_indices
    assert val_indices
    assert train_sources.isdisjoint(val_sources)


def test_early_stopper_triggers_after_patience_without_improvement():
    stopper = train_script.EarlyStopper(patience=2, min_delta=0.01)

    assert stopper.step(1.0) is False
    assert stopper.step(0.95) is False
    assert stopper.step(0.949) is False
    assert stopper.step(0.948) is True
