from pathlib import Path

from concrete_crack_pipeline.config import load_config


def test_config_loads():
    config = load_config(Path(__file__).parents[1] / "config" / "config.json")
    assert config.training.outer_folds == 5
    assert config.analysis.bootstrap_samples == 1000
    assert config.dirs["checkpoints"].name == "checkpoints"
