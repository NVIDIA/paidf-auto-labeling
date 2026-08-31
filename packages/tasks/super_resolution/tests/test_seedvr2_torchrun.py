# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from super_resolution import seedvr2_torchrun


def test_seedvr2_torchrun_rewrites_seedvr_checkpoint_paths(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "seedvr2"

    assert seedvr2_torchrun._rewrite_checkpoint("./ckpts/ema_vae.pth", ckpt_dir) == str(
        ckpt_dir / "ema_vae.pth"
    )
    assert seedvr2_torchrun._rewrite_checkpoint(
        "ckpts/seedvr2_ema_3b.pth",
        ckpt_dir,
    ) == str(ckpt_dir / "seedvr2_ema_3b.pth")
    assert seedvr2_torchrun._rewrite_checkpoint("/models/seedvr2/ema_vae.pth", ckpt_dir) == (
        "/models/seedvr2/ema_vae.pth"
    )
    assert seedvr2_torchrun._rewrite_checkpoint(None, ckpt_dir) is None


def test_seedvr2_torchrun_requires_source_and_checkpoint_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("SEEDVR_ROOT", str(missing))

    with pytest.raises(RuntimeError, match="SEEDVR_ROOT must point"):
        seedvr2_torchrun._require_dir_env("SEEDVR_ROOT")
