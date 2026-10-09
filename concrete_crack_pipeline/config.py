from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DatasetConfig:
    ccic_slug: str = "arnavr10880/concrete-crack-images-for-classification"
    sdnet_slug: str = "harishmulchandani2/sdnet2018"
    ccic_path: str | None = None
    sdnet_path: str | None = None
    source_group_csv: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    base_seed: int = 42
    model_seeds: tuple[int, ...] = (42, 123, 2024)
    tuning_seed: int = 42
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    final_batch_size: int = 32
    use_mixed_precision: bool = True
    mixed_precision_policy: str = "mixed_float16"
    enable_xla: bool = False
    gpu_memory_growth: bool = True
    outer_folds: int = 5
    inner_folds: int = 3
    # Four deliberately spaced configurations: 2 dropout values x 2 L2 values.
    # Tuning is run once per model/outer fold and reused across model seeds.
    tuning_max_configs: int | None = 4
    tuning_epochs: int = 8
    tuning_patience: int = 2
    stage1_epochs: int = 15
    stage2_epochs: int = 20
    stage1_lr: float = 1e-3
    stage2_lr: float = 3e-5
    early_stop_stage1: int = 3
    early_stop_stage2: int = 5
    dropout_values: tuple[float, ...] = (0.2, 0.4)
    l2_values: tuple[float, ...] = (1e-4, 1e-3)


@dataclass(frozen=True)
class AuditConfig:
    remove_exact_duplicates: bool = True
    phash_size: int = 32
    phash_low_frequency_size: int = 8
    phash_threshold: int = 4


@dataclass(frozen=True)
class AnalysisConfig:
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 181
    bootstrap_samples: int = 2000
    gradcam_examples_per_case: int = 3
    latency_samples: int = 100
    run_gradcam: bool = True
    run_latency: bool = True


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path = Path("concrete_crack_research_final")
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def dirs(self) -> dict[str, Path]:
        return {
            "checkpoints": self.output_dir / "checkpoints",
            "histories": self.output_dir / "histories",
            "plots": self.output_dir / "plots",
            "gradcam": self.output_dir / "gradcam",
            "predictions": self.output_dir / "predictions",
            "splits": self.output_dir / "splits",
            "manifests": self.output_dir / "manifests",
            "duplicates": self.output_dir / "duplicates",
            "statistics": self.output_dir / "statistics",
            "external": self.output_dir / "external",
            "errors": self.output_dir / "errors",
            "cache": self.output_dir / "cache",
            "logs": self.output_dir / "logs",
        }

    def ensure_directories(self) -> None:
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def load_config(path: Path | None = None) -> AppConfig:
    if path is None or not path.exists():
        return AppConfig()

    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = DatasetConfig(**raw.get("datasets", {}))
    training_raw = raw.get("training", {})
    if "model_seeds" in training_raw:
        training_raw["model_seeds"] = tuple(training_raw["model_seeds"])
    if "dropout_values" in training_raw:
        training_raw["dropout_values"] = tuple(training_raw["dropout_values"])
    if "l2_values" in training_raw:
        training_raw["l2_values"] = tuple(training_raw["l2_values"])
    training = TrainingConfig(**training_raw)
    audit = AuditConfig(**raw.get("audit", {}))
    analysis = AnalysisConfig(**raw.get("analysis", {}))
    return AppConfig(
        output_dir=Path(raw.get("output_dir", "concrete_crack_research_final")),
        datasets=datasets,
        training=training,
        audit=audit,
        analysis=analysis,
    )
