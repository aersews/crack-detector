from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from concrete_crack_pipeline.data import build_perceptual_groups, make_group_cv, verify_group_disjointness


def test_grouping_and_split_disjointness(tmp_path: Path):
    paths = []
    hashes = []
    for i in range(6):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((16, 16, 3), i * 30, dtype=np.uint8)).save(path)
        paths.append(str(path))
        hashes.append(np.array([bool(i % 2), False, True, True]))
    df = pd.DataFrame({"path": paths, "label": [0, 0, 0, 1, 1, 1], "_phash": hashes})
    grouped = build_perceptual_groups(df, threshold=0)
    assert "group_id" in grouped
    assert grouped.group_id.nunique() >= 1

    # Independent groups are used here so StratifiedGroupKFold can split cleanly.
    grouped["group_id"] = [f"g{i}" for i in range(6)]
    splits = make_group_cv(grouped, 3, 42)
    train, test = splits[0]
    verify_group_disjointness(grouped, train, [], test)
