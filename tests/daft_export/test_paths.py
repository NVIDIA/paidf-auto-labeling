# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from daft_export.paths import ensure_scene_skeleton, resolve_raw_media, scene_paths


class TestScenePaths:
    def test_expected_subpaths(self, tmp_path):
        paths = scene_paths(tmp_path)
        assert paths.scene_dir == tmp_path
        assert paths.raw_dir == tmp_path / "raw"
        assert paths.contextual_dir == tmp_path / "contextual"
        assert paths.contextual_video == tmp_path / "contextual" / "video.json"
        assert paths.contextual_events == tmp_path / "contextual" / "events.json"
        assert paths.contextual_instances == tmp_path / "contextual" / "instances.json"
        assert paths.contextual_objects == tmp_path / "contextual" / "objects.json"
        assert paths.task_dir == tmp_path / "task"
        assert paths.task_mcq == tmp_path / "task" / "mcq.json"
        assert paths.task_bcq == tmp_path / "task" / "bcq.json"
        assert paths.task_open_qa == tmp_path / "task" / "open_qa.json"
        assert paths.sidecars_dir == tmp_path / "sidecars"

    def test_no_filesystem_side_effects(self, tmp_path):
        paths = scene_paths(tmp_path / "new_scene")
        assert not paths.raw_dir.exists()
        assert not paths.contextual_dir.exists()
        assert not paths.task_dir.exists()
        assert not paths.sidecars_dir.exists()


class TestEnsureSceneSkeleton:
    def test_creates_all_subdirs(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path / "scene1")
        assert paths.raw_dir.is_dir()
        assert paths.contextual_dir.is_dir()
        assert paths.task_dir.is_dir()
        assert paths.sidecars_dir.is_dir()

    def test_idempotent(self, tmp_path):
        scene = tmp_path / "scene1"
        ensure_scene_skeleton(scene)
        (scene / "sidecars" / "note.txt").write_text("keep me")
        ensure_scene_skeleton(scene)
        assert (scene / "sidecars" / "note.txt").read_text() == "keep me"


class TestResolveRawMedia:
    def test_returns_none_when_raw_dir_missing(self, tmp_path):
        assert resolve_raw_media(tmp_path / "no-scene") is None

    def test_returns_none_when_empty(self, tmp_path):
        ensure_scene_skeleton(tmp_path)
        assert resolve_raw_media(tmp_path) is None

    def test_resolves_mp4(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path)
        (paths.raw_dir / "clip.mp4").write_bytes(b"\x00")
        assert resolve_raw_media(tmp_path) == paths.raw_dir / "clip.mp4"

    def test_resolves_non_mp4_suffix(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path)
        (paths.raw_dir / "dashcam.mov").write_bytes(b"\x00")
        assert resolve_raw_media(tmp_path) == paths.raw_dir / "dashcam.mov"
