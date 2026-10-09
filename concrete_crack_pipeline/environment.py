from __future__ import annotations

import gc
import json
import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFile


def configure_environment(seed: int, manifest_path: Path, training_cfg=None, logger=None) -> dict:
    """Configure reproducibility and GPU performance before model creation."""
    import tensorflow as tf

    warnings.filterwarnings("default")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    # Determinism is retained for research reproducibility. It is configured
    # before any model/device work below.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    memory_growth_enabled = False
    if gpus and getattr(training_cfg, "gpu_memory_growth", True):
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                memory_growth_enabled = True
            except RuntimeError:
                # Device may already be initialized; this is harmless.
                pass

    mixed_precision_enabled = False
    if gpus and getattr(training_cfg, "use_mixed_precision", True):
        policy_name = getattr(training_cfg, "mixed_precision_policy", "mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy_name)
        mixed_precision_enabled = True
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    xla_enabled = bool(getattr(training_cfg, "enable_xla", False))
    if xla_enabled:
        tf.config.optimizer.set_jit(True)

    manifest = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": bool(tf.test.is_built_with_cuda()),
        "gpus": [d.name for d in gpus],
        "gpu_count": len(gpus),
        "gpu_memory_growth": memory_growth_enabled,
        "mixed_precision": mixed_precision_enabled,
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        "xla": xla_enabled,
        "research_reproducibility": {
            "deterministic_ops_requested": True,
            "note": "Mixed precision can produce tiny floating-point differences; final metrics must be reported from the saved run."
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if logger:
        logger.info("GPU devices: %s", manifest["gpus"] or "none")
        logger.info("Mixed precision: %s (%s)", mixed_precision_enabled, manifest["mixed_precision_policy"])
        logger.info("GPU memory growth: %s | XLA: %s", memory_growth_enabled, xla_enabled)
    return manifest


def cleanup_tensorflow_memory() -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()
