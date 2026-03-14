import numpy as np
import torch

import scripts.merge_data as merge_data_script
import scripts.train as train_script
from scripts.play import OnlineGameData
from scripts.self_play import save_dataset


def test_online_game_data_save_persists_game_ids(tmp_path):
    data = OnlineGameData()
    data.samples = [
        {
            "board": np.zeros((15, 10, 9), dtype=np.float32),
            "policy": np.zeros((8010,), dtype=np.float32),
            "value": 0.25,
            "game_id": "online-game-1",
        }
    ]

    output_path = tmp_path / "online_test.pt"
    data.save(str(output_path))

    saved = torch.load(output_path, weights_only=False)

    assert saved["game_ids"] == ["online-game-1"]


def test_selfplay_save_dataset_persists_game_ids(tmp_path):
    save_dataset(
        data=[
            {
                "board": np.zeros((15, 10, 9), dtype=np.float32),
                "policy": np.zeros((8010,), dtype=np.float32),
                "value": 0.5,
                "game_id": "selfplay-game-1",
            }
        ],
        save_dir=str(tmp_path),
        suffix="test",
    )

    saved_files = list(tmp_path.glob("selfplay_test_*.pt"))
    assert len(saved_files) == 1

    saved = torch.load(saved_files[0], weights_only=False)

    assert saved["game_ids"] == ["selfplay-game-1"]


def test_merge_data_files_preserves_game_ids(tmp_path):
    sample = {
        "boards": torch.zeros((2, 15, 10, 9), dtype=torch.float32),
        "policies": torch.zeros((2, 8010), dtype=torch.float32),
        "values": torch.tensor([0.1, -0.2], dtype=torch.float32),
        "game_ids": ["g1", "g2"],
    }
    input_path = tmp_path / "online_test.pt"
    output_path = tmp_path / "merged_test.pt"
    torch.save(sample, input_path)

    merge_data_script.merge_data_files([str(input_path)], str(output_path))

    merged = torch.load(output_path, weights_only=False)

    assert merged["game_ids"] == ["g1", "g2"]


def test_dataset_loads_game_ids_and_split_prefers_game_groups(tmp_path):
    sample = {
        "boards": torch.zeros((4, 15, 10, 9), dtype=torch.float32),
        "policies": torch.zeros((4, 8010), dtype=torch.float32),
        "values": torch.tensor([0.2, 0.1, -0.1, -0.2], dtype=torch.float32),
        "game_ids": ["g1", "g1", "g2", "g2"],
    }
    torch.save(sample, tmp_path / "merged_test.pt")

    dataset = train_script.AlphaZeroDataset(str(tmp_path))
    assert [entry["game_id"] for entry in dataset.samples] == ["g1", "g1", "g2", "g2"]

    train_indices, val_indices, grouped = train_script.split_indices_by_source(
        dataset.samples,
        val_split=0.5,
        seed=42,
    )

    train_game_ids = {dataset.samples[i]["game_id"] for i in train_indices}
    val_game_ids = {dataset.samples[i]["game_id"] for i in val_indices}

    assert grouped is True
    assert train_game_ids.isdisjoint(val_game_ids)
