from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import CheckpointStore
from .config import load_config
from .pipeline import Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research-grade concrete crack CNN comparison pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/config.json"))
    parser.add_argument("--reset-progress", action="store_true", help="Reset pipeline checkpoints before starting")
    parser.add_argument("--show-status", action="store_true", help="Show checkpoint status and exit")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.ensure_directories()
    store = CheckpointStore(config.dirs["checkpoints"] / "pipeline_state.json")
    if args.show_status:
        print(json.dumps(store.state, indent=2, default=str))
        return 0
    if args.reset_progress:
        store.reset()
    Pipeline(config).run()
    return 0
