# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from core import read_pipeline_state
from visual_qa_service.main import VisualQaService, build_config, main


def test_build_config_maps_endpoint_args(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text('{"questions": []}\n', encoding="utf-8")
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    args = parser.parse_args(
        [
            "--generation-mode",
            "window-vlm-llm",
            "--question-bank-file",
            str(bank_path),
            "--image-group-dir",
            "/data/identity",
            "--max-group-images",
            "5",
            "--vlm-endpoint-url",
            "http://vlm.test/v1",
            "--llm-endpoint-url",
            "http://llm.test/v1",
            "--vlm-model",
            "vlm-test",
            "--llm-model",
            "llm-test",
        ]
    )

    config = build_config(args)

    assert config.vlm_endpoint_url == "http://vlm.test/v1"
    assert config.llm_endpoint_url == "http://llm.test/v1"
    assert config.vlm_model == "vlm-test"
    assert config.llm_model == "llm-test"
    assert config.image_group_dir == "/data/identity"
    assert config.max_group_images == 5
    assert not hasattr(args, "vlm_api_key_env")
    assert not hasattr(args, "llm_api_key_env")


def test_build_config_maps_no_flat_qa_tasks(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    default_config = build_config(parser.parse_args([]))
    assert default_config.emit_flat_qa_tasks is True

    disabled_config = build_config(parser.parse_args(["--no-flat-qa-tasks"]))
    assert disabled_config.emit_flat_qa_tasks is False


def test_build_config_image_group_defaults() -> None:
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    config = build_config(parser.parse_args([]))

    assert config.image_group_dir is None
    assert config.max_group_images == 0


def test_build_config_parser_defaults_to_instruct() -> None:
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    assert build_config(parser.parse_args([])).parser == "instruct"


def test_build_config_maps_parser_arg() -> None:
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    args = parser.parse_args(["--parser", "reasoning"])

    assert build_config(args).parser == "reasoning"


def test_build_config_supports_legacy_question_driven_mode() -> None:
    parser = argparse.ArgumentParser()
    VisualQaService().add_service_args(parser)

    args = parser.parse_args(
        [
            "--generation-mode",
            "question-driven-vlm-llm",
            "--question-bank-file",
            "bank.json",
            "--question-prompt-max-tokens",
            "4096",
        ]
    )
    config = build_config(args)

    assert config.generation_mode == "question-driven-vlm-llm"
    assert config.question_prompt_max_tokens == 4096


def test_visual_qa_service_writes_normalized_sidecar(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip_001"
    (scene_dir / "sidecars" / "visual_qa").mkdir(parents=True)

    bank_path = tmp_path / "bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "q1",
                    "question": "Is the road occupied?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    _write_json(
        scene_dir / "sidecars" / "visual_qa" / "windows.json",
        {
            "windows": [
                {
                    "items": [
                        {
                            "id": "q1",
                            "question": "Is the road occupied?",
                            "options": ["Yes", "No"],
                            "answer": "A",
                        }
                    ]
                }
            ]
        },
    )
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    argv = [
        "visual-qa-service",
        "--input-file",
        str(input_file),
        "--question-bank-file",
        str(bank_path),
    ]
    with patch.object(sys, "argv", argv):
        main()

    payload = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    assert payload["media_id"] == "clip_001"
    assert payload["items"][0]["answer"] == "Yes"
    bcq = json.loads((scene_dir / "task" / "bcq.json").read_text(encoding="utf-8"))
    assert not (scene_dir / "task" / "mcq.json").exists()
    assert bcq["items"][0]["video_id"] == "clip_001"
    assert bcq["items"][0]["answer"] == "Yes"
    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert outcomes["mcq"].success is False
    assert outcomes["bcq"].success is True
    assert outcomes["open_qa"].success is False


def test_container_uses_controlled_media_toolchain() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    # FFmpeg and the OpenCV/PyAV wheels come from the shared media base, so the
    # service installs the prebuilt wheels instead of compiling them.
    assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
    assert "ffmpeg-install" not in text
    assert "media_toolchain.py python-install" not in text
    assert "--no-install-package av" in text
    assert "--no-install-package opencv-python" in text
    assert "--no-install-package opencv-python-headless" in text
    assert '--find-links "${UPA_MEDIA_WHEEL_DIR}"' in text
    assert "opencv-python-headless av" in text
    assert text.count("media_toolchain.py verify") == 2
    assert "apt-get install -y --no-install-recommends ffmpeg" not in text


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
