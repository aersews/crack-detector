import json
from pathlib import Path

from concrete_crack_pipeline.checkpoint import CheckpointStore


def test_checkpoint_round_trip_and_atomic_state(tmp_path: Path):
    path = tmp_path / "checkpoints" / "state.json"
    store = CheckpointStore(path)
    store.mark_completed("item/1", artifact="result.json")
    store.mark_failed("item/2", "temporary failure")

    restored = CheckpointStore(path)
    assert restored.is_completed("item/1")
    assert restored.get_completed("item/1")["artifact"] == "result.json"
    assert restored.state["failed"]["item/2"]["error"] == "temporary failure"
    assert json.loads(path.read_text())["schema_version"] == 1
