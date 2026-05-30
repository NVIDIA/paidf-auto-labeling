# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import config.loader
import config.schema
import pipeline


def _repo_root() -> Path:
    # tests/ -> repo root
    return Path(__file__).resolve().parents[2]


def test_sr_disabled_does_not_route_tracking_or_window_to_sr(tmp_path: Path, caplog) -> None:
    """
    When super_resolution.enabled=false, the input video (not an SR path) should be
    referenced in the dry-run log.  SR stage must not appear at all.
    """
    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True)
    video_path = in_dir / "clip.mp4"
    video_path.write_bytes(b"")

    config_path = _repo_root() / "configs" / "pipeline_example.yaml"
    overrides = [
        f"data.0.inputs.video_path={video_path}",
        f"data.0.output.out_dir={tmp_path / 'out'}",
        f"data.0.output.log_dir={tmp_path / 'out' / 'logs'}",
        "endpoints.vlm.url=http://example.invalid/v1",
        "endpoints.vlm.model=dummy-vlm",
        "endpoints.llm.url=http://example.invalid/v1",
        "endpoints.llm.model=dummy-llm",
        "super_resolution.enabled=false",
    ]

    cfg, cfg_dir = config.loader.load_config_with_overrides(str(config_path), overrides, logger=None)
    validated = config.schema.validate_schema(cfg, logger=None)
    assert validated is not None

    sample = validated.data[0]

    caplog.set_level(logging.INFO)
    rc = pipeline.run_pipeline(
        sample,
        validated,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=cfg_dir,
        repo_root=_repo_root(),
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "out" / "logs",
        dry_run=True,
    )
    assert rc == 0

    # SR is disabled: SR stage should not appear at all.
    assert "[sr]" not in caplog.text

    # SR is disabled: no SR output paths should be referenced in logs.
    assert "super_resolution_video.mp4" not in caplog.text

    # The real input video must be referenced (in the [inputs] DRY RUN line).
    assert str(video_path) in caplog.text
