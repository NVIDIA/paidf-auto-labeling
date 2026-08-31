# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import json
from importlib.metadata import requires

import pytest
import regex as re
from pydantic import ValidationError
from reasoning.common import (
    DAFT_VERSION,
    DaftConvertError,
    SceneContext,
    daft_envelope,
    metadata_block,
    write_daft_json,
)
from reasoning.text import re_search_timeout


class TestRuntimeDependencies:
    def test_regex_is_declared_runtime_dependency(self):
        requirements = requires("reasoning") or []
        assert "regex>=2026.1.15" in requirements


class TestVersionLiteral:
    def test_is_metropolis_v3_0(self):
        # nvidia-tao-daft v3 schemas pin ``version`` to this exact string;
        # any drift here means strict ``tao-daft validate`` will reject every
        # file we produce.
        assert DAFT_VERSION == "metropolis-v3.0"


class TestDaftConvertError:
    def test_subclasses_value_error(self):
        # Downstream can catch ValueError as a broader net.
        assert issubclass(DaftConvertError, ValueError)


class TestSceneContext:
    def test_video_scene_defaults(self):
        ctx = SceneContext(media_id="clip01")
        assert ctx.media_id == "clip01"
        assert ctx.is_image is False
        assert ctx.scene_id_field == "video_id"
        assert ctx.license_str is None
        assert ctx.tags == ()

    def test_image_scene_uses_image_id_field(self):
        ctx = SceneContext(media_id="frame01", is_image=True)
        assert ctx.scene_id_field == "image_id"

    def test_iso_date_default_today(self):
        ctx = SceneContext(media_id="x")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ctx.iso_date)

    def test_iso_date_overridable(self):
        ctx = SceneContext(media_id="x", iso_date="2026-04-20")
        assert ctx.iso_date == "2026-04-20"

    def test_from_input_video(self, tmp_path):
        p = tmp_path / "main.mp4"
        p.write_bytes(b"\x00")
        ctx = SceneContext.from_input(p)
        assert ctx.media_id == "main"
        assert ctx.is_image is False

    def test_from_input_image(self, tmp_path):
        p = tmp_path / "frame.jpg"
        p.write_bytes(b"\x00")
        ctx = SceneContext.from_input(p)
        assert ctx.media_id == "frame"
        assert ctx.is_image is True

    def test_from_input_overrides(self, tmp_path):
        p = tmp_path / "main.mp4"
        p.write_bytes(b"\x00")
        ctx = SceneContext.from_input(p, license_str="MIT", tags=("traffic",))
        assert ctx.license_str == "MIT"
        assert ctx.tags == ("traffic",)

    def test_frozen(self):
        # Frozen so the per-scene context can't drift mid-pipeline; if a
        # caller wants to override a field, they explicitly build a new ctx.
        ctx = SceneContext(media_id="x")
        with pytest.raises(ValidationError, match="frozen"):
            ctx.media_id = "y"  # type: ignore[misc]


class TestMetadataBlock:
    def test_minimum(self):
        block = metadata_block("mcq")
        assert block["type"] == "mcq"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", block["date"], timeout=re_search_timeout())
        assert set(block.keys()) == {"type", "date"}

    def test_explicit_date(self):
        block = metadata_block("bcq", iso_date="2026-04-20")
        assert block["date"] == "2026-04-20"

    def test_with_optional_fields(self):
        block = metadata_block(
            "mcq",
            iso_date="2026-04-20",
            description="trial run",
            license_str="MIT",
            tags=["traffic", "accident"],
        )
        assert block == {
            "type": "mcq",
            "date": "2026-04-20",
            "description": "trial run",
            "license": "MIT",
            "tags": ["traffic", "accident"],
        }

    def test_omits_unset_optionals(self):
        # DAFT uses `additionalProperties: false`, so absent keys must stay absent.
        block = metadata_block("mcq", iso_date="2026-04-20")
        assert "description" not in block
        assert "license" not in block
        assert "tags" not in block


class TestDaftEnvelope:
    def test_video_scene_envelope(self):
        ctx = SceneContext(media_id="main", iso_date="2026-04-20")
        env = daft_envelope("video", ctx)
        assert env["version"] == DAFT_VERSION
        assert env["video_id"] == "main"
        assert env["metadata"] == {"type": "video", "date": "2026-04-20"}

    def test_image_scene_envelope(self):
        ctx = SceneContext(media_id="frame", is_image=True, iso_date="2026-04-20")
        env = daft_envelope("image", ctx)
        assert env["image_id"] == "frame"
        assert "video_id" not in env

    def test_omits_scene_id_when_disabled(self):
        # instances.json / mcq.json / bcq.json don't carry a top-level scene id.
        ctx = SceneContext(media_id="main")
        env = daft_envelope("instances", ctx, include_scene_id=False)
        assert "video_id" not in env
        assert "image_id" not in env
        assert env["metadata"]["type"] == "instances"

    def test_threads_license_and_tags_from_ctx(self):
        ctx = SceneContext(
            media_id="x",
            iso_date="2026-04-20",
            license_str="MIT",
            tags=("traffic", "accident"),
        )
        env = daft_envelope("video", ctx)
        assert env["metadata"]["license"] == "MIT"
        assert env["metadata"]["tags"] == ["traffic", "accident"]

    def test_description_passthrough(self):
        ctx = SceneContext(media_id="x", iso_date="2026-04-20")
        env = daft_envelope("video", ctx, description="trial run")
        assert env["metadata"]["description"] == "trial run"


class TestWriteDaftJson:
    def test_writes_payload(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"version": DAFT_VERSION, "metadata": {"type": "mcq"}, "items": [{"x": 1}]}
        write_daft_json(path, payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "out.json"
        write_daft_json(
            path,
            {"version": DAFT_VERSION, "metadata": {"type": "mcq"}, "items": [{"x": 1}]},
        )
        assert path.exists()

    def test_rejects_non_dict(self, tmp_path):
        with pytest.raises(ValueError, match="must be a dict"):
            write_daft_json(tmp_path / "x.json", ["not", "a", "dict"])

    def test_rejects_missing_version(self, tmp_path):
        with pytest.raises(ValueError, match="version"):
            write_daft_json(tmp_path / "x.json", {"metadata": {"type": "mcq"}, "items": [{"x": 1}]})

    def test_rejects_wrong_version(self, tmp_path):
        # The shape guard exists exactly to catch this kind of bug before
        # it surfaces as a cryptic tao-daft validator error.
        with pytest.raises(ValueError, match="metropolis-v3.0"):
            write_daft_json(
                tmp_path / "x.json",
                {"version": "3.0", "metadata": {"type": "mcq"}, "items": [{"x": 1}]},
            )

    def test_rejects_missing_metadata(self, tmp_path):
        with pytest.raises(ValueError, match="metadata"):
            write_daft_json(tmp_path / "x.json", {"version": DAFT_VERSION})

    def test_rejects_missing_metadata_type(self, tmp_path):
        with pytest.raises(ValueError, match="metadata.type"):
            write_daft_json(tmp_path / "x.json", {"version": DAFT_VERSION, "metadata": {}})

    def test_rejects_unexpected_metadata_type(self, tmp_path):
        with pytest.raises(ValueError, match="expected"):
            write_daft_json(
                tmp_path / "x.json",
                {"version": DAFT_VERSION, "metadata": {"type": "mcq"}, "items": [{"x": 1}]},
                expected_type="open_qa",
            )
