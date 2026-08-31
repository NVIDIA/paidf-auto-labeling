# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from core import DataEntry, ScenePaths, read_pipeline_state
from PIL import Image
from super_resolution import SuperResolutionConfig, SuperResolutionTask
from super_resolution.resolver import Resolver, SrResult
from super_resolution.task import MediaResolution, probe_media_resolution


class _FakeResolver(Resolver):
    def __init__(self) -> None:
        self.seen_media_path: Path | None = None

    def run(self, media_path: Path, scene_paths: ScenePaths) -> SrResult:
        self.seen_media_path = media_path
        output_path = scene_paths.sidecars_dir / f"sr_output{media_path.suffix.lower()}"
        output_path.write_bytes(b"enhanced")
        return SrResult(success=True, output_path=output_path)


class _MissingOutputResolver(Resolver):
    def run(self, media_path: Path, scene_paths: ScenePaths) -> SrResult:
        output_path = scene_paths.sidecars_dir / f"sr_output{media_path.suffix.lower()}"
        return SrResult(success=True, output_path=output_path)


def test_super_resolution_task_records_pipeline_state(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    resolver = _FakeResolver()

    entry = DataEntry(media_path=str(media), data_path=str(scene_dir))
    out = SuperResolutionTask(resolver=resolver).run(entry)

    assert out == entry.model_copy(
        update={"media_path": str(scene_dir / "sidecars" / "sr_output.mp4")}
    )
    assert resolver.seen_media_path == media

    state = read_pipeline_state(scene_dir)
    assert state.data_entry_id == entry.id
    assert state.media_path == str(media)
    assert state.enhanced_media is not None
    assert state.enhanced_media.success is True
    assert state.enhanced_media.output_path == str(scene_dir / "sidecars" / "sr_output.mp4")


def test_auto_resolution_policy_skips_high_resolution_input(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    resolver = _FakeResolver()
    entry = DataEntry(media_path=str(media), data_path=str(scene_dir))
    config = SuperResolutionConfig(resolution_policy="auto")

    out = SuperResolutionTask(
        config=config,
        resolver=resolver,
        resolution_probe=lambda _path: MediaResolution(width=1920, height=1080),
    ).run(entry)

    assert out == entry
    assert resolver.seen_media_path is None
    state = read_pipeline_state(scene_dir)
    assert state.enhanced_media is not None
    assert state.enhanced_media.success is False
    assert state.enhanced_media.output_path is None


def test_auto_resolution_policy_runs_low_resolution_input(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    resolver = _FakeResolver()
    entry = DataEntry(media_path=str(media), data_path=str(scene_dir))
    config = SuperResolutionConfig(resolution_policy="auto")

    out = SuperResolutionTask(
        config=config,
        resolver=resolver,
        resolution_probe=lambda _path: MediaResolution(width=640, height=360),
    ).run(entry)

    assert out == entry.model_copy(
        update={"media_path": str(scene_dir / "sidecars" / "sr_output.mp4")}
    )
    assert resolver.seen_media_path == media
    state = read_pipeline_state(scene_dir)
    assert state.enhanced_media is not None
    assert state.enhanced_media.success is True
    assert state.enhanced_media.output_path == str(scene_dir / "sidecars" / "sr_output.mp4")


def test_auto_resolution_policy_runs_when_probe_fails(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    resolver = _FakeResolver()
    entry = DataEntry(media_path=str(media), data_path=str(scene_dir))
    config = SuperResolutionConfig(resolution_policy="auto")

    def fail_probe(_path: Path) -> MediaResolution:
        raise RuntimeError("probe failed")

    SuperResolutionTask(
        config=config,
        resolver=resolver,
        resolution_probe=fail_probe,
    ).run(entry)

    assert resolver.seen_media_path == media


def test_super_resolution_task_does_not_promote_missing_success_output(
    tmp_path: Path,
) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    entry = DataEntry(media_path=str(media), data_path=str(scene_dir))

    out = SuperResolutionTask(resolver=_MissingOutputResolver()).run(entry)

    missing_output = scene_dir / "sidecars" / "sr_output.mp4"
    assert not missing_output.exists()
    assert out == entry.model_copy(update={"media_path": entry.media_path})
    state = read_pipeline_state(scene_dir)
    assert state.enhanced_media is not None
    assert state.enhanced_media.success is False
    assert state.enhanced_media.output_path == str(missing_output)


def test_disabled_task_does_not_construct_or_run_resolver(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    config = SuperResolutionConfig(enabled=False)

    SuperResolutionTask(config=config).run(
        DataEntry(media_path=str(media), data_path=str(scene_dir))
    )

    assert not scene_dir.exists()


def test_probe_media_resolution_reads_image_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 240)).save(image_path)

    assert probe_media_resolution(image_path) == MediaResolution(width=320, height=240)
