# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mcq_generation.mcq.utils.validate import (
    is_empty_marker_valid,
    is_metadata_valid,
    is_task_file_valid,
    outputs_complete,
)


def test_is_task_file_valid(tmp_path):
    p = tmp_path / "mcq.json"

    assert not is_task_file_valid(p)

    p.write_text("invalid")
    assert not is_task_file_valid(p)

    p.write_text("[]")
    assert not is_task_file_valid(p)

    p.write_text("{}")
    assert not is_task_file_valid(p)

    p.write_text('{"items": []}')
    assert not is_task_file_valid(p)

    p.write_text('{"items": [{"question": "q", "answer": "A"}]}')
    assert is_task_file_valid(p)


def test_is_empty_marker_valid(tmp_path):
    p = tmp_path / "mcq.empty.json"

    assert not is_empty_marker_valid(p)

    p.write_text("invalid")
    assert not is_empty_marker_valid(p)

    p.write_text("{}")
    assert not is_empty_marker_valid(p)

    p.write_text('{"mcq": [], "_error": "invalid_empty_marker"}')
    assert not is_empty_marker_valid(p)

    p.write_text('{"mcq": [{"question": "q"}], "_error": "zero_task_items"}')
    assert not is_empty_marker_valid(p)

    p.write_text('{"mcq": [], "_error": "zero_task_items"}')
    assert is_empty_marker_valid(p)


def test_is_metadata_valid(tmp_path):
    p = tmp_path / "meta.json"

    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    p.write_text("{}")
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    p.write_text('{"windows": []}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Window missing frame keys
    p.write_text('{"windows": [{"start_frame": 0}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Window missing enhanced key
    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Window missing caption key (when direct=False)
    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": "val"}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Enhanced payload must be an MCQ object, not a string or unrelated object.
    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": "val", "cap": "val"}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": {}, "cap": "val"}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": {"version": 2.0}, "cap": "val"}]}')
    assert not is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Valid (direct=False) — both caption and enhanced object present
    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": {"mcq": []}, "cap": "val"}]}')
    assert is_metadata_valid(p, direct_mcq_from_vlm=False, caption_key="cap", enhanced_caption_key="enh")

    # Valid (direct=True) — only enhanced object needed
    p.write_text('{"windows": [{"start_frame": 0, "end_frame": 10, "enh": {"mcq": []}}]}')
    assert is_metadata_valid(p, direct_mcq_from_vlm=True, caption_key="cap", enhanced_caption_key="enh")


def _write_scene(scene_dir, *, mcq=True, bcq=False, open_qa=False, empty_stub=False):
    (scene_dir / "sidecars").mkdir(parents=True, exist_ok=True)
    (scene_dir / "task").mkdir(parents=True, exist_ok=True)
    (scene_dir / "sidecars" / "metadata.json").write_text(
        '{"windows": [{"start_frame": 0, "end_frame": 1, "enh": {"mcq": []}, "cap": "v"}]}'
    )
    if mcq:
        (scene_dir / "task" / "mcq.json").write_text(
            '{"version": "metropolis-v3.0", "items": [{"question": "q", "answer": "A"}]}'
        )
    if bcq:
        (scene_dir / "task" / "bcq.json").write_text(
            '{"version": "metropolis-v3.0", "items": [{"question": "q", "answer": "Yes"}]}'
        )
    if open_qa:
        (scene_dir / "task" / "open_qa.json").write_text(
            '{"version": "metropolis-v3.0", "items": [{"question": "q", "answer": "free text"}]}'
        )
    if empty_stub:
        (scene_dir / "sidecars" / "mcq.empty.json").write_text('{"mcq": [], "_error": "zero_task_items"}')


def test_outputs_complete_accepts_mcq_only_scene(tmp_path):
    _write_scene(tmp_path, mcq=True)
    assert outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )


def test_outputs_complete_accepts_bcq_only_scene(tmp_path):
    _write_scene(tmp_path, mcq=False, bcq=True)
    assert outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )


def test_outputs_complete_accepts_open_qa_only_scene(tmp_path):
    _write_scene(tmp_path, mcq=False, open_qa=True)
    assert outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )


def test_outputs_complete_accepts_empty_stub_as_finished(tmp_path):
    _write_scene(tmp_path, mcq=False, empty_stub=True)
    assert outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )


def test_outputs_complete_rejects_stale_empty_stub(tmp_path):
    _write_scene(tmp_path, mcq=False)
    (tmp_path / "sidecars" / "mcq.empty.json").write_text('{"mcq": [], "_error": "invalid_empty_marker"}')
    assert not outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )


def test_outputs_complete_rejects_missing_task_output(tmp_path):
    _write_scene(tmp_path, mcq=False)
    assert not outputs_complete(
        out_dir=tmp_path,
        direct_mcq_from_vlm=False,
        caption_key="cap",
        enhanced_caption_key="enh",
    )
