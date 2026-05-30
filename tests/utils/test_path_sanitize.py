# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from al_utils.io import write_json
from al_utils.path_sanitize import sanitize_paths_for_publish
from config.normalize import resolve_input_path, resolve_path


def test_sanitize_paths_for_publish_rewrites_absolute_paths() -> None:
    artifact_root = Path("/artifacts/auto-labeling")
    payload = {
        "input_media_path": "/artifacts/auto-labeling/augmentation/data/sample/input.png",
        "output_media_path": "/artifacts/auto-labeling/augmentation/output/sample/output.png",
        "container_path": "/app/data/sample/output.png",
        "remote": "s3://bucket/path/video.mp4",
        "caption": "not a path",
    }

    sanitized = sanitize_paths_for_publish(payload, artifact_root=artifact_root)

    assert sanitized["input_media_path"] == "{artifact_root}/augmentation/data/sample/input.png"
    assert sanitized["output_media_path"] == "{artifact_root}/augmentation/output/sample/output.png"
    assert sanitized["container_path"] == "{artifact_root}/sample/output.png"
    assert sanitized["remote"] == "s3://bucket/path/video.mp4"
    assert sanitized["caption"] == "not a path"


def test_write_json_can_sanitize_paths(tmp_path: Path) -> None:
    out = tmp_path / "metadata.json"
    write_json(
        out,
        {"source_video": "/workspace/input/sample_video.mp4"},
        sanitize_paths=True,
        artifact_root="/workspace",
    )

    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["source_video"] == "{artifact_root}/input/sample_video.mp4"
    assert "/workspace" not in out.read_text(encoding="utf-8")


def test_artifact_root_token_resolves_at_read_time(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    media = artifact_root / "input" / "sample.mp4"
    media.parent.mkdir()
    media.write_bytes(b"")
    monkeypatch.setenv("AUTO_LABELING_ARTIFACT_ROOT", str(artifact_root))

    repo_root = tmp_path / "repo"
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)

    assert resolve_path("{artifact_root}/input/sample.mp4", config_dir=config_dir, repo_root=repo_root) == str(media)
    assert resolve_input_path("{artifact_root}/input/sample.mp4", config_dir=config_dir, repo_root=repo_root) == str(
        media
    )


def test_artifact_root_token_falls_back_to_repo_root_when_file_exists(tmp_path: Path) -> None:
    repo_root = tmp_path / "auto-labeling"
    config_dir = repo_root / "configs"
    media = repo_root / "input" / "sample.mp4"
    media.parent.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    media.write_bytes(b"")

    assert resolve_input_path("{artifact_root}/input/sample.mp4", config_dir=config_dir, repo_root=repo_root) == str(
        media
    )
