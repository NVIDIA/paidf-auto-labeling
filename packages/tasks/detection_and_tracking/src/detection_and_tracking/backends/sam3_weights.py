# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve local SAM3 weight layouts for Transformers and native Meta runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from detection_and_tracking.backends.runtime import resolve_model_cache_root
from detection_and_tracking.config import DetectionAndTrackingConfig

Sam3Version = Literal["sam3", "sam3.1"]
Sam3RuntimeKind = Literal["transformers", "native"]

_NATIVE_CHECKPOINT_NAMES: dict[Sam3Version, tuple[str, ...]] = {
    "sam3": ("sam3.pt", "sam3_video.pt"),
    "sam3.1": ("sam3.1_multiplex.pt", "sam3.1.pt"),
}


@dataclass(frozen=True)
class Sam3WeightPaths:
    """Resolved local weight locations for a SAM3 run."""

    root: Path
    runtime: Sam3RuntimeKind
    version: Sam3Version
    transformers_dir: Path | None = None
    native_checkpoint: Path | None = None

    @property
    def path_for_runtime(self) -> Path:
        if self.runtime == "transformers":
            if self.transformers_dir is None:
                raise FileNotFoundError(
                    "SAM3 Transformers weights missing. Mount an HF-format directory "
                    "at /models/sam3 or set SAM3_MODEL_PATH to that directory."
                )
            return self.transformers_dir
        if self.native_checkpoint is None:
            expected = " or ".join(_NATIVE_CHECKPOINT_NAMES[self.version])
            raise FileNotFoundError(
                "SAM3 native checkpoint missing. Mount "
                f"{expected} under /models/sam3 or set SAM3_MODEL_PATH to the .pt file."
            )
        return self.native_checkpoint


def resolve_sam3_runtime(
    config: DetectionAndTrackingConfig,
) -> Sam3RuntimeKind:
    """Select Transformers vs native Meta runtime from config."""
    requested = config.sam3_runtime
    version = config.sam3_version
    if requested == "transformers":
        if version == "sam3.1":
            raise ValueError(
                "sam3_version='sam3.1' requires sam3_runtime='native' (or 'auto'). "
                "Hugging Face Transformers does not ship SAM 3.1 Object Multiplex."
            )
        return "transformers"
    if requested == "native":
        return "native"
    # auto
    return "native" if version == "sam3.1" else "transformers"


def resolve_sam3_weights(
    config: DetectionAndTrackingConfig,
    *,
    runtime: Sam3RuntimeKind | None = None,
) -> Sam3WeightPaths:
    """Resolve local SAM3 weights without downloading."""
    selected_runtime = runtime or resolve_sam3_runtime(config)
    version = config.sam3_version
    root = _resolve_sam3_root(config)
    transformers_dir = _find_transformers_dir(root)
    native_checkpoint = _find_native_checkpoint(root, version)
    paths = Sam3WeightPaths(
        root=root,
        runtime=selected_runtime,
        version=version,
        transformers_dir=transformers_dir,
        native_checkpoint=native_checkpoint,
    )
    # Validate the selected runtime has usable weights.
    _ = paths.path_for_runtime
    return paths


def _resolve_sam3_root(config: DetectionAndTrackingConfig) -> Path:
    explicit = os.getenv("SAM3_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return resolve_model_cache_root(config.model_cache_path) / "sam3"


def _find_transformers_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    if root.is_file():
        return None
    # HF layout: config.json next to model shards / safetensors.
    if (root / "config.json").is_file():
        return root
    # Nested HF export, e.g. /models/sam3/facebook-sam3/
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child
    return None


def _find_native_checkpoint(root: Path, version: Sam3Version) -> Path | None:
    if not root.exists():
        return None
    if root.is_file():
        return root if root.suffix == ".pt" else None
    for name in _NATIVE_CHECKPOINT_NAMES[version]:
        candidate = root / name
        if candidate.is_file():
            return candidate
    # Allow a single .pt under the directory when the name is custom.
    pts = sorted(path for path in root.glob("*.pt") if path.is_file())
    if len(pts) == 1:
        return pts[0]
    return None


__all__ = [
    "Sam3RuntimeKind",
    "Sam3Version",
    "Sam3WeightPaths",
    "resolve_sam3_runtime",
    "resolve_sam3_weights",
]
