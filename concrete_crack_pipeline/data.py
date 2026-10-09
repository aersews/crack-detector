from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except (OSError, ValueError):
        return False


def perceptual_hash(path: Path, hash_size: int = 32, low_frequency_size: int = 8) -> np.ndarray:
    with Image.open(path) as img:
        pixels = np.asarray(
            img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    coefficients = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coefficients[:low_frequency_size, :low_frequency_size]
    values = low.flatten()
    return values > np.median(values[1:])


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def build_perceptual_groups(df: pd.DataFrame, threshold: int, logger: Callable[[str], None] = print) -> pd.DataFrame:
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = df["_phash"].tolist()
    logger(f"Building perceptual groups for {n:,} images...")
    edge_count = 0
    for i in range(n):
        if i and i % 1000 == 0:
            logger(f"  Similarity scan: {i:,}/{n:,}")
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                union(i, j)
                edge_count += 1

    roots = [find(i) for i in range(n)]
    root_to_group: dict[int, str] = {}
    group_ids = []
    for root in roots:
        root_to_group.setdefault(root, f"PHASH_{len(root_to_group):06d}")
        group_ids.append(root_to_group[root])

    result = df.copy()
    result["group_id"] = group_ids
    mixed = result.groupby("group_id")["label"].nunique()
    mixed_count = int((mixed > 1).sum())
    logger(f"Perceptual similarity edges: {edge_count:,}")
    logger(f"Similarity groups: {result['group_id'].nunique():,}")
    if mixed_count:
        logger(f"WARNING: {mixed_count} groups contain both positive and negative labels.")
    return result


def resolve_dataset_path(configured_path: str | None, kaggle_slug: str) -> Path:
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {path}")
        return path
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required when a local dataset path is not configured.") from exc
    return Path(kagglehub.dataset_download(kaggle_slug))


def find_class_images(dataset_path: Path, class_name: str) -> list[Path]:
    class_dir = dataset_path / class_name
    if not class_dir.exists():
        return []
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_ccic_dataset(dataset_path: Path) -> pd.DataFrame:
    positive = find_class_images(dataset_path, "Positive") or find_class_images(dataset_path, "positive")
    negative = find_class_images(dataset_path, "Negative") or find_class_images(dataset_path, "negative")
    if not positive or not negative:
        raise RuntimeError("Could not find Positive/Negative CCIC image directories.")
    rows = ([{"path": str(p), "label": 1, "class": "Positive", "dataset": "CCIC"} for p in positive] +
            [{"path": str(p), "label": 0, "class": "Negative", "dataset": "CCIC"} for p in negative])
    return pd.DataFrame(rows)


def audit_images(df: pd.DataFrame, output_path: Path, logger: Callable[[str], None] = print) -> pd.DataFrame:
    valid, dimensions, modes = [], [], []
    for i, path_str in enumerate(df["path"]):
        if i and i % 2000 == 0:
            logger(f"  Image audit: {i:,}/{len(df):,}")
        path = Path(path_str)
        ok = verify_image(path)
        valid.append(ok)
        if ok:
            with Image.open(path) as img:
                dimensions.append(f"{img.width}x{img.height}")
                modes.append(img.mode)
        else:
            dimensions.append(None)
            modes.append(None)

    result = df.copy()
    result["valid"] = valid
    result["dimensions"] = dimensions
    result["mode"] = modes
    invalid = result[~result["valid"]].copy()
    if not invalid.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        invalid.to_csv(output_path, index=False)
        logger(f"WARNING: removing {len(invalid):,} invalid images.")
    return result[result["valid"]].drop(columns=["valid"]).reset_index(drop=True)


def add_exact_hashes(df: pd.DataFrame, logger: Callable[[str], None] = print) -> pd.DataFrame:
    hashes = []
    for i, path_str in enumerate(df["path"]):
        if i and i % 2000 == 0:
            logger(f"  SHA-256: {i:,}/{len(df):,}")
        hashes.append(file_sha256(Path(path_str)))
    result = df.copy()
    result["sha256"] = hashes
    return result


def audit_exact_duplicates(df: pd.DataFrame, output_path: Path) -> list[dict]:
    rows = []
    for sha, group in df.groupby("sha256"):
        if len(group) <= 1:
            continue
        labels = group["label"].unique().tolist()
        rows.append({"sha256": sha, "count": len(group), "labels": labels, "paths": group["path"].tolist()})
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if any(len(row["labels"]) > 1 for row in rows):
        raise RuntimeError("Identical image content occurs under multiple labels.")
    return rows


def remove_exact_duplicates(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    if not enabled:
        return df
    return df.sort_values(["sha256", "path"]).drop_duplicates("sha256", keep="first").reset_index(drop=True)


def add_perceptual_hashes(df: pd.DataFrame, hash_size: int, low_frequency_size: int) -> pd.DataFrame:
    result = df.copy()
    result["_phash"] = [perceptual_hash(Path(p), hash_size, low_frequency_size) for p in result["path"]]
    return result


def apply_true_source_groups(df: pd.DataFrame, source_group_csv: str) -> pd.DataFrame:
    mapping = pd.read_csv(source_group_csv)
    required = {"path", "source_group"}
    if not required.issubset(mapping.columns):
        raise ValueError("source_group_csv must contain 'path' and 'source_group'.")
    mapping["_absolute_path"] = mapping["path"].astype(str).map(lambda x: str(Path(x).expanduser().resolve()))
    result = df.copy()
    result["_absolute_path"] = result["path"].map(lambda x: str(Path(x).expanduser().resolve()))
    result = result.merge(mapping[["_absolute_path", "source_group"]], on="_absolute_path", how="left")
    missing = int(result["source_group"].isna().sum())
    if missing:
        raise ValueError(f"{missing} images are missing source-group IDs.")
    result["group_id"] = result["source_group"].astype(str)
    return result.drop(columns=["_absolute_path", "source_group"])


def create_dataset_groups(df: pd.DataFrame, source_group_csv: str | None, threshold: int, logger=print) -> pd.DataFrame:
    if source_group_csv:
        return apply_true_source_groups(df, source_group_csv)
    return build_perceptual_groups(df, threshold, logger)


def save_dataset_manifest(df: pd.DataFrame, output_dir: Path, phash_threshold: int, using_true_groups: bool) -> dict:
    group_sizes = df.groupby("group_id").size()
    mixed_groups = df.groupby("group_id")["label"].nunique()
    manifest = {
        "n_images": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "n_groups": int(df["group_id"].nunique()),
        "group_size_mean": float(group_sizes.mean()),
        "group_size_median": float(group_sizes.median()),
        "group_size_max": int(group_sizes.max()),
        "mixed_label_groups": int((mixed_groups > 1).sum()),
        "phash_threshold": None if using_true_groups else phash_threshold,
        "grouping_method": "true_source_group" if using_true_groups else "perceptual_hash_connected_components",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    df.drop(columns=["_phash"], errors="ignore").to_csv(output_dir / "dataset_index.csv", index=False)
    return manifest


def make_group_cv(df: pd.DataFrame, n_splits: int, seed: int):
    if df["group_id"].nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct groups, found {df['group_id'].nunique()}.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros(len(df))
    return list(cv.split(X, df["label"].to_numpy(), df["group_id"].to_numpy()))


def verify_group_disjointness(df: pd.DataFrame, train_idx, validation_idx, test_idx) -> None:
    train = set(df.iloc[train_idx]["group_id"])
    validation = set(df.iloc[validation_idx]["group_id"])
    test = set(df.iloc[test_idx]["group_id"])
    if train & validation or train & test or validation & test:
        raise RuntimeError("Group leakage detected between train, validation, and test splits.")


def save_split(df: pd.DataFrame, train_idx, validation_idx, test_idx, output_path: Path) -> None:
    rows = []
    for split_name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
        for idx in indices:
            row = df.iloc[idx]
            rows.append({"path": row["path"], "label": int(row["label"]), "group_id": row["group_id"], "split": split_name})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def find_sdnet_images(root: Path) -> pd.DataFrame:
    surface_map = {"d": "BridgeDeck", "w": "Wall", "p": "Pavement"}
    rows = []
    label_pattern = re.compile(r"(?:^|[_-])(cracked|uncracked|c|u)(?:$|[_-])", re.I)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = [p.lower() for p in path.parts]
        surface = next((val for key, val in surface_map.items() if key in parts), "Unknown")
        label = None
        candidates = [*parts, path.stem.lower()]
        for token in candidates:
            match = label_pattern.search(token)
            if match:
                label = 1 if match.group(1) in {"c", "cracked"} else 0
                break
        if label is not None:
            rows.append({"path": str(path), "label": label, "surface": surface, "dataset": "SDNET2018"})
    if not rows:
        raise RuntimeError("No SDNET2018 images were identified. Check the directory structure.")
    return pd.DataFrame(rows)


def prepare_external_dataset(root: Path, dirs, audit_cfg, logger=print) -> pd.DataFrame:
    df = find_sdnet_images(root)
    logger("SDNET2018 label counts:\n" + str(df["label"].value_counts().sort_index()))
    df = audit_images(df, dirs["manifests"] / "invalid_sdnet_images.csv", logger)
    df = add_exact_hashes(df, logger)
    df = remove_exact_duplicates(df, audit_cfg.remove_exact_duplicates)
    df = add_perceptual_hashes(df, audit_cfg.phash_size, audit_cfg.phash_low_frequency_size)
    df = build_perceptual_groups(df, audit_cfg.phash_threshold, logger)
    return df.drop(columns=["_phash"]).reset_index(drop=True)
