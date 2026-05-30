# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests: load the blueprint config with overrides and run dry-run planning."""

from __future__ import annotations

from pathlib import Path

import config.loader
import config.schema
import pipeline
import pytest

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PIPELINE_EXAMPLE_CONFIG = CONFIGS_DIR / "pipeline_example.yaml"

# The repo intentionally ships a single blueprint config. These scenarios verify that we can
# express common "preset" behaviors via CLI overrides.
SCENARIOS: list[tuple[str, list[str]]] = [
    ("full_pipeline_like", []),
    (
        "tracking_only_like",
        [
            "super_resolution.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=false",
            "detection_and_tracking.enabled=true",
        ],
    ),
    (
        "vlm_json_only_like",
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=true",
            "mcq_generation.enabled=false",
        ],
    ),
    (
        "window_direct_vlm_like",
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=true",
            "mcq_generation.mode=window-direct-vlm",
        ],
    ),
    (
        "window_vlm_llm_like",
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=true",
            "mcq_generation.mode=window-vlm-llm",
        ],
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_overrides(tmp_path: Path, video_path: Path) -> list[str]:
    return [
        f"data.0.inputs.video_path={video_path}",
        f"data.0.output.out_dir={tmp_path / 'out'}",
        f"data.0.output.log_dir={tmp_path / 'out' / 'logs'}",
        "endpoints.vlm.url=http://example.invalid/v1",
        "endpoints.vlm.model=dummy-vlm",
        "endpoints.llm.url=http://example.invalid/v1",
        "endpoints.llm.model=dummy-llm",
    ]


@pytest.mark.parametrize(("scenario_name", "scenario_overrides"), SCENARIOS)
def test_config_load_validate_normalize_dry_run(
    scenario_name: str, scenario_overrides: list[str], tmp_path: Path
) -> None:
    """Load config, validate schema, and run pipeline with dry_run=True."""
    if not PIPELINE_EXAMPLE_CONFIG.exists():
        pytest.skip(f"Config not found: {PIPELINE_EXAMPLE_CONFIG}")

    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True)
    dummy_video = in_dir / "dummy.mp4"
    dummy_video.write_bytes(b"")

    overrides = _base_overrides(tmp_path, dummy_video) + scenario_overrides

    config_obj, config_dir = config.loader.load_config_with_overrides(
        str(PIPELINE_EXAMPLE_CONFIG), overrides, logger=None
    )
    assert config_obj is not None
    assert isinstance(config_dir, Path)

    validated = config.schema.validate_schema(config_obj, logger=None)
    assert validated is not None, f"Schema validation failed for scenario={scenario_name}"

    sample = validated.data[0]

    rc = pipeline.run_pipeline(
        sample,
        validated,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=config_dir,
        repo_root=_repo_root(),
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "out" / "logs",
        dry_run=True,
    )
    assert rc == 0, f"Pipeline dry-run failed for scenario={scenario_name} (rc={rc})"
