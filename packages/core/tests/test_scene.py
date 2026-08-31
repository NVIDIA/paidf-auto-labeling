# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pytest
from core import (
    CONTEXTUAL_DIRNAME,
    RAW_DIRNAME,
    RAW_MEDIA_BASENAME,
    SIDECARS_DIRNAME,
    TASK_DIRNAME,
    DataEntry,
    SceneContext,
    ScenePipelineState,
    active_media_path_for_input,
    ensure_scene_skeleton,
    find_active_media_path,
    is_image_path,
    raw_analyzed_media_name,
    raw_media_path,
    resolve_sidecar,
    scene_context_for_entry,
    stage_raw_media,
    write_pipeline_state,
)
from pydantic import ValidationError


def test_resolve_sidecar_returns_contained_path(tmp_path: Path) -> None:
    root = tmp_path / "sidecars"
    resolved = resolve_sidecar(root, "visual_qa/items.json")
    assert resolved == (root / "visual_qa" / "items.json").resolve(strict=False)


@pytest.mark.parametrize(
    "relative_name",
    ["../escape.json", "../../etc/passwd", "visual_qa/../../escape.json", "/abs/secret"],
)
def test_resolve_sidecar_rejects_escapes(tmp_path: Path, relative_name: str) -> None:
    with pytest.raises(ValueError, match="escapes scene sidecars root"):
        resolve_sidecar(tmp_path / "sidecars", relative_name)


def test_image_extension_detection() -> None:
    assert is_image_path("clip.PNG") is True
    assert is_image_path(Path("clip.mp4")) is False


def test_scene_context_from_video() -> None:
    ctx = SceneContext.from_input("/tmp/abc.mp4")
    assert ctx.media_id == "abc"
    assert ctx.is_image is False
    assert ctx.scene_id_field == "video_id"


def test_scene_context_from_image() -> None:
    ctx = SceneContext.from_input("/tmp/abc.JPG")
    assert ctx.is_image is True
    assert ctx.scene_id_field == "image_id"


def test_scene_context_for_entry_uses_input_media_stem(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()

    ctx = scene_context_for_entry(
        DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))
    )

    assert ctx.media_id == "clip"
    assert ctx.is_image is False


def test_scene_context_for_entry_uses_scene_id_for_active_media(tmp_path: Path) -> None:
    scene = tmp_path / "clip_001"
    active = active_media_path_for_input(scene, "frame.jpg")
    active.parent.mkdir(parents=True)
    active.touch()

    ctx = scene_context_for_entry(DataEntry(media_path=str(active), data_path=str(scene)))

    assert ctx.media_id == "clip_001"
    assert ctx.is_image is True


def test_scene_context_for_entry_prefers_persisted_media_id_for_active_media(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "pipeline_data_x"
    active = active_media_path_for_input(scene, "frame.jpg")
    active.parent.mkdir(parents=True)
    active.touch()
    write_pipeline_state(scene, ScenePipelineState(media_id="clip_001"))

    ctx = scene_context_for_entry(DataEntry(media_path=str(active), data_path=str(scene)))

    assert ctx.media_id == "clip_001"
    assert ctx.is_image is True


def test_scene_context_for_entry_uses_scene_id_for_transient_sidecar_media(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "clip_001"
    sr_output = scene / SIDECARS_DIRNAME / "sr_output.mp4"
    sr_output.parent.mkdir(parents=True)
    sr_output.touch()

    ctx = scene_context_for_entry(DataEntry(media_path=str(sr_output), data_path=str(scene)))

    assert ctx.media_id == "clip_001"
    assert ctx.is_image is False
    assert ctx.scene_id_field == "video_id"


def test_scene_context_for_entry_prefers_persisted_media_id_for_transient_sidecar_media(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "pipeline_data_x"
    sr_output = scene / SIDECARS_DIRNAME / "sr_output.mp4"
    sr_output.parent.mkdir(parents=True)
    sr_output.touch()
    write_pipeline_state(scene, ScenePipelineState(media_id="clip_001"))

    ctx = scene_context_for_entry(DataEntry(media_path=str(sr_output), data_path=str(scene)))

    assert ctx.media_id == "clip_001"
    assert ctx.is_image is False


def test_scene_context_for_entry_uses_scene_id_for_raw_media(tmp_path: Path) -> None:
    scene = tmp_path / "clip_001"
    raw = raw_media_path(scene, "clip.mp4")
    raw.parent.mkdir(parents=True)
    raw.touch()

    ctx = scene_context_for_entry(DataEntry(media_path=str(raw), data_path=str(scene)))

    assert ctx.media_id == "clip_001"
    assert ctx.is_image is False


def test_scene_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SceneContext.model_validate({"media_id": "clip", "unexpected": True})


def test_ensure_scene_skeleton_creates_dirs(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    for d in (paths.raw_dir, paths.contextual_dir, paths.task_dir, paths.sidecars_dir):
        assert d.is_dir()


def test_active_media_path_for_input_uses_source_suffix(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    sidecars_dir = scene_dir / "sidecars"

    assert active_media_path_for_input(scene_dir, "clip.mp4") == sidecars_dir / "active.mp4"
    assert active_media_path_for_input(scene_dir, "frame.png") == sidecars_dir / "active.png"
    assert active_media_path_for_input(scene_dir, "frame.JPG") == sidecars_dir / "active.JPG"


def test_active_media_path_for_input_requires_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must include a file suffix"):
        active_media_path_for_input(tmp_path / "scene", "clip")


def test_raw_media_path_uses_video_or_source_image_suffix(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"

    assert raw_media_path(scene_dir, "clip.mp4") == scene_dir / "sidecars" / "raw.mp4"
    assert raw_media_path(scene_dir, "frame.png") == scene_dir / "sidecars" / "raw.png"
    assert raw_media_path(scene_dir, "frame.JPG") == scene_dir / "sidecars" / "raw.JPG"
    assert RAW_MEDIA_BASENAME == "raw"


def test_find_active_media_path_prefers_video_then_image(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    sidecars_dir = scene_dir / "sidecars"
    sidecars_dir.mkdir(parents=True)
    image_active = sidecars_dir / "active.png"
    video_active = sidecars_dir / "active.mp4"
    image_active.write_bytes(b"image")

    assert find_active_media_path(scene_dir) == image_active

    video_active.write_bytes(b"video")
    assert find_active_media_path(scene_dir) == video_active


def test_find_active_media_path_accepts_preserved_case_image_suffix(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    sidecars_dir = scene_dir / "sidecars"
    sidecars_dir.mkdir(parents=True)
    image_active = sidecars_dir / "active.JPG"
    image_active.write_bytes(b"image")

    assert find_active_media_path(scene_dir) == image_active


def test_find_active_media_path_ignores_non_media_sidecars(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    sidecars_dir = scene_dir / "sidecars"
    sidecars_dir.mkdir(parents=True)
    active_metadata = sidecars_dir / "active.json"
    active_metadata.write_text("{}", encoding="utf-8")

    assert find_active_media_path(scene_dir) is None


def test_scene_paths_match_layout(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    assert paths.scene_dir == tmp_path / "scene"
    assert paths.raw_dir.name == RAW_DIRNAME
    assert paths.contextual_dir.name == CONTEXTUAL_DIRNAME
    assert paths.task_dir.name == TASK_DIRNAME
    assert paths.sidecars_dir.name == SIDECARS_DIRNAME


def test_raw_analyzed_media_name_appends_missing_suffix() -> None:
    assert raw_analyzed_media_name("clip", ".mp4") == "clip.mp4"


def test_raw_analyzed_media_name_avoids_doubled_suffix() -> None:
    # Scene media ids carry the extension (scene dir "clip.mp4"); the suffix must
    # not be appended again, which would produce "clip.mp4.mp4".
    assert raw_analyzed_media_name("clip.mp4", ".mp4") == "clip.mp4"
    assert raw_analyzed_media_name("clip.MP4", ".mp4") == "clip.MP4"


def test_stage_raw_media_symlinks_single_extension(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    source = paths.sidecars_dir / "active.mp4"
    source.write_bytes(b"video")

    staged = stage_raw_media(
        raw_dir=paths.raw_dir,
        media_path=source,
        media_id="clip.mp4",
        copy_media=False,
        logger=logging.getLogger("core.tests.test_scene"),
    )

    assert staged == paths.raw_dir / "clip.mp4"
    assert staged.is_symlink()
    assert staged.read_bytes() == b"video"
    # Exactly one raw entry, with a single extension (regression: .mp4.mp4).
    assert [p.name for p in paths.raw_dir.iterdir()] == ["clip.mp4"]


def test_stage_raw_media_copy_is_self_contained(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    source = tmp_path / "external.mp4"
    source.write_bytes(b"payload")

    staged = stage_raw_media(
        raw_dir=paths.raw_dir,
        media_path=source,
        media_id="external.mp4",
        copy_media=True,
        logger=logging.getLogger("core.tests.test_scene"),
    )

    assert staged == paths.raw_dir / "external.mp4"
    assert not staged.is_symlink()
    assert staged.read_bytes() == b"payload"


def test_stage_raw_media_copy_short_circuits_same_path(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    source = paths.raw_dir / "clip.mp4"
    source.write_bytes(b"already staged")

    staged = stage_raw_media(
        raw_dir=paths.raw_dir,
        media_path=source,
        media_id="clip.mp4",
        copy_media=True,
        logger=logging.getLogger("core.tests.test_scene"),
    )

    assert staged == source
    assert source.read_bytes() == b"already staged"


def test_stage_raw_media_missing_source_returns_none(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")

    staged = stage_raw_media(
        raw_dir=paths.raw_dir,
        media_path=tmp_path / "missing.mp4",
        media_id="missing.mp4",
        copy_media=False,
        logger=logging.getLogger("core.tests.test_scene"),
    )

    assert staged is None
    assert list(paths.raw_dir.iterdir()) == []
