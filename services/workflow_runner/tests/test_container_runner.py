# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from core import DataEntry, EmptyOutputPolicy, StagePolicyError
from workflow_runner.container_runner import (
    ContainerPipelineRunner,
    ContainerRunnerConfig,
    ContainerStage,
    VolumeMount,
    build_container_plan,
    normalize_entries_for_container,
    parse_stage_args,
    parse_volume_mount,
    resolve_container_user,
    rootless_env_with_defaults,
    runner_config_from_args,
    write_runner_input_file,
)
from workflow_runner.cookbook import apply_cookbook_config, expand_directory_entries
from workflow_runner.main import WorkflowRunnerService, _apply_container_defaults, main


def _runner_args() -> argparse.Namespace:
    args = WorkflowRunnerService().build_parser().parse_args([])
    args.stages = ["super_resolution", "detection_and_tracking", "captioning", "reasoning"]
    args.policy = "fail"
    args.vlm_endpoint_url = "http://vlm:8000/v1"
    args.vlm_model = "vlm-model"
    args.llm_endpoint_url = "http://llm:8000/v1"
    args.llm_model = "llm-model"
    args.container_dry_run = True
    args.cookbook_root = None
    _apply_container_defaults(args)
    return args


@contextmanager
def _runner_input_file(entries: list[DataEntry]) -> Iterator[Path]:
    input_file, tempdir = write_runner_input_file(entries)
    try:
        yield input_file
    finally:
        tempdir.cleanup()


def _stage_arg_values(raw_args: list[str], stage: str) -> list[str]:
    prefix = f"{stage}="
    return [raw_arg.removeprefix(prefix) for raw_arg in raw_args if raw_arg.startswith(prefix)]


def _stage_flag_values(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    index = 0
    while index < len(values):
        value = values[index]
        if not value.startswith("--"):
            index += 1
            continue
        flag, separator, inline = value.partition("=")
        if separator:
            parsed[flag] = inline
            index += 1
            continue
        if index + 1 < len(values) and not values[index + 1].startswith("--"):
            parsed[flag] = values[index + 1]
            index += 2
            continue
        parsed[flag] = ""
        index += 1
    return parsed


def test_container_plan_matches_pending_stage_service_contracts(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media = media_dir / "clip.mp4"
    media.touch()
    data_dir = tmp_path / "data" / "scene"
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(data_dir))]
    )
    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(_runner_args(), entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == [
        "super_resolution",
        "detection_and_tracking",
        "captioning",
        "reasoning",
    ]
    assert [stage.image for stage in plan.stages] == [
        "paidf-super-resolution-service",
        "paidf-detection-and-tracking-rfdetr-service",
        "paidf-captioning-service",
        "paidf-reasoning-service",
    ]
    for stage in plan.stages:
        assert "--input-file" in stage.args
        assert str(input_file) in stage.args
    assert any(mount.source == media.parent and mount.read_only for mount in plan.mounts)
    assert any(mount.source == data_dir and not mount.read_only for mount in plan.mounts)


def test_container_plan_orders_explicit_stages_by_canonical_dataflow(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["reasoning", "visual_qa", "captioning"]

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == ["captioning", "visual_qa", "reasoning"]


def test_container_plan_can_include_visual_qa_and_stage_args(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    question_bank = tmp_path / "question_bank.json"
    question_bank.write_text('{"questions": []}\n', encoding="utf-8")
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["captioning", "visual_qa", "reasoning"]
    args.question_bank_file = str(question_bank)
    args.stage_arg = [
        "captioning=--window-seconds",
        "captioning=15",
        "visual_qa=--generation-mode",
        "visual_qa=window-vlm-llm",
    ]

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == ["captioning", "visual_qa", "reasoning"]
    captioning = plan.stages[0]
    visual_qa = plan.stages[1]
    assert captioning.args[-2:] == ("--window-seconds", "15")
    assert "--enable-visual-qa" not in captioning.args
    assert "--question-bank-file" in visual_qa.args
    assert str(question_bank) in visual_qa.args
    assert "--vlm-endpoint-url" in visual_qa.args
    assert args.vlm_endpoint_url in visual_qa.args
    assert "--llm-endpoint-url" in visual_qa.args
    assert args.llm_endpoint_url in visual_qa.args
    assert visual_qa.args[-2:] == ("--generation-mode", "window-vlm-llm")
    assert any(mount.source == question_bank.parent and mount.read_only for mount in plan.mounts)


def test_container_runner_dry_run_builds_docker_command(tmp_path: Path) -> None:
    mount = VolumeMount(source=tmp_path, target=str(tmp_path))
    log_path = tmp_path / "scene" / "logs" / "workflow_runner.jsonl"
    config = ContainerRunnerConfig(
        mounts=(mount,),
        dry_run=True,
        user="1000:1000",
        env=("NVIDIA_API_KEY",),
    )
    runner = ContainerPipelineRunner(
        config=config,
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="captioning",
        image="captioning-service",
        build_target="captioning-service:main",
        args=("--input-file", str(tmp_path / "input.jsonl")),
        log_paths=(log_path,),
    )

    with patch("workflow_runner.container_runner.subprocess.run") as subprocess_run:
        results = runner.run([stage])

    subprocess_run.assert_not_called()
    assert not log_path.exists()
    assert results[0].skipped is True
    assert results[0].command[:3] == ("docker", "run", "--rm")
    assert "--gpus" in results[0].command
    assert "--user" in results[0].command
    assert "1000:1000" in results[0].command
    assert "--mount" in results[0].command
    assert "captioning-service" in results[0].command


def test_container_runner_rejects_unknown_runtime(tmp_path: Path) -> None:
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(runtime=str(tmp_path / "fake-runtime"), dry_run=True),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="captioning",
        image="captioning-service",
        build_target="captioning-service:main",
        args=("--input-file", str(tmp_path / "input.jsonl")),
    )

    try:
        runner.stage_command(stage)
    except ValueError as exc:
        assert "Unsupported container runtime" in str(exc)
    else:
        raise AssertionError("stage_command should reject unknown container runtimes")


def test_runner_config_omits_disabled_gpu_and_network() -> None:
    args = _runner_args()
    args.container_gpus = "none"
    args.container_network = "none"
    args.container_user = "none"

    config = runner_config_from_args(args, ())

    assert config.gpus is None
    assert config.network is None
    assert config.user is None


def test_runner_config_passes_container_user() -> None:
    args = _runner_args()
    args.container_user = "1000:1000"

    config = runner_config_from_args(args, ())

    assert config.user == "1000:1000"


def test_runner_config_rejects_unknown_runtime() -> None:
    args = _runner_args()
    args.container_runtime = "bash"

    try:
        runner_config_from_args(args, ())
    except ValueError as exc:
        assert "Unsupported container runtime" in str(exc)
    else:
        raise AssertionError("runner_config_from_args should reject unknown container runtimes")


def test_container_runner_writes_per_scene_event_log(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    scene = tmp_path / "scene"
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(scene))]
    )
    args = _runner_args()
    args.stages = ["captioning"]
    completed: subprocess.CompletedProcess[list[str]] = subprocess.CompletedProcess(
        args=[],
        returncode=0,
    )

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)
        runner = ContainerPipelineRunner(
            config=ContainerRunnerConfig(mounts=plan.mounts),
            policy=EmptyOutputPolicy.FAIL,
            logger=logging.getLogger("test"),
        )
        with patch("workflow_runner.container_runner.subprocess.run", return_value=completed):
            results = runner.run(list(plan.stages))

    log_path = scene.resolve() / "logs" / "workflow_runner.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert results[0].success is True
    assert results[0].node_id == "captioning"
    assert [event["event"] for event in events] == ["start", "success"]
    assert [event["node_id"] for event in events] == ["captioning", "captioning"]
    assert [event["stage"] for event in events] == ["captioning", "captioning"]
    assert events[0]["image"] == "paidf-captioning-service"
    assert events[0]["command"][:2] == ["docker", "run"]
    assert events[1]["returncode"] == 0


def test_container_runner_writes_failure_event_before_policy_error(tmp_path: Path) -> None:
    log_path = tmp_path / "scene" / "logs" / "workflow_runner.jsonl"
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="captioning",
        image="captioning-service",
        build_target="captioning-service:main",
        args=("--input-file", str(tmp_path / "input.jsonl")),
        log_paths=(log_path,),
    )
    completed: subprocess.CompletedProcess[list[str]] = subprocess.CompletedProcess(
        args=[],
        returncode=12,
    )

    with (
        pytest.raises(StagePolicyError, match="captioning failed"),
        patch("workflow_runner.container_runner.subprocess.run", return_value=completed),
    ):
        runner.run([stage])

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["start", "failure"]
    assert events[1]["returncode"] == 12


def test_runner_input_file_tempdir_can_be_cleaned_up(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]

    input_file, tempdir = write_runner_input_file(entries)

    assert input_file.exists()
    tempdir.cleanup()
    assert not input_file.exists()


def test_runner_input_file_is_container_readable(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]

    input_file, tempdir = write_runner_input_file(entries)

    try:
        assert stat.S_IMODE(input_file.parent.stat().st_mode) == 0o755
        assert stat.S_IMODE(input_file.stat().st_mode) == 0o644
    finally:
        tempdir.cleanup()


def test_parse_volume_mount_rejects_empty_host() -> None:
    try:
        parse_volume_mount(":/target:ro")
    except ValueError as exc:
        assert "host path must not be empty" in str(exc)
    else:
        raise AssertionError("parse_volume_mount should reject an empty host path")


def test_parse_volume_mount_rejects_relative_explicit_target(tmp_path: Path) -> None:
    try:
        parse_volume_mount(f"{tmp_path}:relative-target:ro")
    except ValueError as exc:
        assert "explicit container target must be absolute" in str(exc)
    else:
        raise AssertionError("parse_volume_mount should reject a relative explicit target")


def test_build_stage_image_uses_absolute_build_script(tmp_path: Path) -> None:
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(build_images=True),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="captioning",
        image="captioning-service",
        build_target="captioning-service:main",
        args=("--input-file", str(tmp_path / "input.jsonl")),
    )
    completed: subprocess.CompletedProcess[list[str]] = subprocess.CompletedProcess(
        args=[],
        returncode=0,
    )

    with patch("workflow_runner.container_runner.subprocess.run", return_value=completed) as run:
        runner.run([stage])

    build_command = run.call_args_list[0].args[0]
    build_script = next((arg for arg in build_command if arg.endswith("scripts/build.py")), None)
    assert build_script is not None
    assert Path(build_script).is_absolute()


def _build_targets_built(run_mock: MagicMock) -> list[str]:
    """Return build targets passed to scripts/build.py across mocked runs."""
    return [
        call.args[0][-1]
        for call in run_mock.call_args_list
        if any(str(arg).endswith("scripts/build.py") for arg in call.args[0])
    ]


def test_container_ensure_images_flag_sets_config() -> None:
    args = WorkflowRunnerService().build_parser().parse_args(["--container-ensure-images"])

    assert args.container_ensure_images is True

    config = runner_config_from_args(args, mounts=())
    assert config.ensure_images is True
    assert config.build_images is False


def test_ensure_images_builds_only_missing_images(tmp_path: Path) -> None:
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(ensure_images=True, remove=False),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    present = ContainerStage(
        name="visual_qa",
        image="visual-qa-service",
        build_target="visual-qa-service:build",
        args=("--input-file", str(tmp_path / "in.jsonl")),
    )
    missing = ContainerStage(
        name="person_attribute_search",
        image="event-and-person-attribute-search-service",
        build_target="event-and-person-attribute-search-service:build",
        args=("--input-file", str(tmp_path / "in.jsonl")),
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[list[str]]:
        if "inspect" in command:
            # visual-qa-service is present (0); the PAS image is missing (1).
            returncode = 0 if command[-1] == "visual-qa-service" else 1
            return subprocess.CompletedProcess(args=command, returncode=returncode)
        return subprocess.CompletedProcess(args=command, returncode=0)

    with patch("workflow_runner.container_runner.subprocess.run", side_effect=fake_run) as run:
        runner.run([present, missing])

    assert _build_targets_built(run) == ["event-and-person-attribute-search-service:build"]


def test_build_images_forces_rebuild_regardless_of_presence(tmp_path: Path) -> None:
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(build_images=True, ensure_images=True, remove=False),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="person_attribute_search",
        image="event-and-person-attribute-search-service",
        build_target="event-and-person-attribute-search-service:build",
        args=("--input-file", str(tmp_path / "in.jsonl")),
    )

    completed: subprocess.CompletedProcess[list[str]] = subprocess.CompletedProcess(
        args=[],
        returncode=0,
    )

    with patch("workflow_runner.container_runner.subprocess.run", return_value=completed) as run:
        runner.run([stage])

    # Force build short-circuits the existence probe entirely.
    assert not any("inspect" in call.args[0] for call in run.call_args_list)
    assert _build_targets_built(run) == ["event-and-person-attribute-search-service:build"]


def test_build_images_requires_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ContainerPipelineRunner(
        config=ContainerRunnerConfig(build_images=True),
        policy=EmptyOutputPolicy.FAIL,
        logger=logging.getLogger("test"),
    )
    stage = ContainerStage(
        name="captioning",
        image="captioning-service",
        build_target="captioning-service:main",
        args=("--input-file", str(tmp_path / "input.jsonl")),
    )
    monkeypatch.setattr(
        "workflow_runner.container_runner._build_script_path",
        lambda: tmp_path / "missing" / "scripts" / "build.py",
    )

    with pytest.raises(StagePolicyError, match="requires a source checkout"):
        runner.run([stage])


def test_stage_service_args_resolve_local_paths(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["super_resolution", "visual_qa", "reasoning", "training_export"]
    args.model_cache_path = "ckpts"
    args.question_bank_file = "question_bank.json"
    args.reasoning_config_file = "reasoning.yaml"
    args.training_export_formats = ["cosmos-reason-v1.0", "tao-vl-reason-v1.0"]
    args.training_export_dir = "exports/training"
    args.training_export_tasks = ["mcq"]
    args.training_export_description = "workflow export"
    args.training_export_license = "internal"
    args.training_export_tags = ["traffic"]
    args.training_export_no_copy_media = True
    args.training_export_emit_media_root_as_null = True

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    sr_args = plan.stages[0].args
    visual_qa_args = plan.stages[1].args
    reasoning_args = plan.stages[2].args
    training_export_args = plan.stages[3].args
    assert str(Path("ckpts").resolve()) in sr_args
    assert str(Path("question_bank.json").resolve()) in visual_qa_args
    assert str(Path("reasoning.yaml").resolve()) in reasoning_args
    assert plan.stages[3].name == "training_export"
    assert plan.stages[3].image == "paidf-training-export-service"
    assert _stage_flag_values(list(training_export_args))["--training-export-dir"] == str(
        Path("exports/training").resolve()
    )
    assert training_export_args.count("--training-export-format") == 2
    assert "--training-export-no-copy-media" in training_export_args
    assert "--training-export-emit-media-root-as-null" in training_export_args
    export_dir = Path("exports/training").resolve()
    assert any(mount.source == export_dir and not mount.read_only for mount in plan.mounts)


def test_training_export_dir_mount_requires_training_export_stage(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    export_dir = tmp_path / "exports"
    cache_dir = tmp_path / "cache"
    args = _runner_args()
    args.stages = ["captioning"]
    args.model_cache_path = str(cache_dir)
    args.training_export_formats = ["cosmos-reason-v1.0"]
    args.training_export_dir = str(export_dir)

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert not export_dir.exists()
    assert any(mount.source == cache_dir and not mount.read_only for mount in plan.mounts)
    assert all(mount.source != export_dir for mount in plan.mounts)


def test_stage_arg_rejects_runner_owned_flags() -> None:
    with pytest.raises(ValueError, match="--input-file"):
        parse_stage_args(("captioning=--input-file=/reserved-input.jsonl",))


def test_build_container_plan_rejects_reserved_extra_args(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["captioning"]
    args.stage_arg = ["captioning=--log-level"]

    with _runner_input_file(entries) as input_file:
        with pytest.raises(ValueError, match="--log-level"):
            build_container_plan(args, entries, input_file=input_file)


def test_cookbook_config_selects_image_pipeline_and_expands_directory(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "images"
    media_dir.mkdir()
    first = media_dir / "first.jpg"
    second = media_dir / "second.png"
    ignored = media_dir / "notes.txt"
    first.touch()
    second.touch()
    ignored.touch()
    output_dir = tmp_path / "out"
    question_bank = tmp_path / "question_bank.json"
    question_bank.write_text('{"questions": []}\n', encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Describe the image.\n", encoding="utf-8")
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        f"""
pipeline: image
runtime:
  model_cache_path: {tmp_path / "ckpts"}
  gpu_ids: "0"
data:
  - inputs:
      media_path: {media_dir}
    id: image-batch
    output:
      out_dir: {output_dir}
endpoints:
  vlm:
    url: http://vlm:8000/v1
    model: vlm-model
captioning:
  enabled: true
detection_and_tracking:
  enabled: false
visual_qa:
  enabled: true
  question_bank_file: {question_bank}
reasoning:
  enabled: true
stage_args:
  captioning:
    - --image-prompt-file
    - {prompt}
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})
    expanded = expand_directory_entries(entries, pipeline=args.pipeline)

    assert args.pipeline == "image"
    assert args.stages == ["captioning", "visual_qa", "reasoning"]
    assert args.vlm_endpoint_url == "http://vlm:8000/v1"
    assert args.vlm_model == "vlm-model"
    assert args.question_bank_file == str(question_bank)
    assert args.model_cache_path == str(tmp_path / "ckpts")
    assert args.gpu_ids == "0"
    assert args.stage_arg == ["captioning=--image-prompt-file", f"captioning={prompt}"]
    assert entries[0].id == "image-batch"
    assert [entry.id for entry in expanded] == ["image-batch:first.jpg", "image-batch:second.png"]
    assert [Path(entry.media_path).name for entry in expanded] == ["first.jpg", "second.png"]
    assert [Path(entry.data_path).name for entry in expanded] == ["first.jpg", "second.png"]


def test_expand_directory_entries_uses_collision_free_output_names(tmp_path: Path) -> None:
    media_dir = tmp_path / "images"
    media_dir.mkdir()
    (media_dir / "frame.jpg").touch()
    (media_dir / "frame.png").touch()
    entries = [
        DataEntry(
            id="frames",
            media_path=str(media_dir),
            data_path=str(tmp_path / "out"),
        )
    ]

    expanded = expand_directory_entries(entries, pipeline="image")

    assert [entry.id for entry in expanded] == ["frames:frame.jpg", "frames:frame.png"]
    assert [Path(entry.data_path).name for entry in expanded] == ["frame.jpg", "frame.png"]


def test_expand_directory_entries_uses_filename_prefix_when_id_is_empty(tmp_path: Path) -> None:
    media_dir = tmp_path / "images"
    media_dir.mkdir()
    (media_dir / "frame.jpg").touch()
    entries = [
        DataEntry(
            id="",
            media_path=str(media_dir),
            data_path=str(tmp_path / "out"),
        )
    ]

    expanded = expand_directory_entries(entries, pipeline="image")

    assert [entry.id for entry in expanded] == ["frame.jpg:frame.jpg"]


def test_cookbook_stage_arg_paths_resolve_relative_to_config(tmp_path: Path) -> None:
    cookbook_root = tmp_path / "default_image"
    config_dir = cookbook_root / "configs"
    prompt_dir = cookbook_root / "prompts" / "captioning"
    config_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    media = tmp_path / "image.jpg"
    media.touch()
    image_prompt = prompt_dir / "image_prompt.md"
    summary_prompt = prompt_dir / "summary_prompt.md"
    image_prompt.write_text("Describe this image.\n", encoding="utf-8")
    summary_prompt.write_text("Summarize descriptions.\n", encoding="utf-8")
    config = config_dir / "pipeline_image.yaml"
    config.write_text(
        f"""
pipeline: image
data:
  - inputs:
      media_path: {media}
    output:
      out_dir: {tmp_path / "out"}
captioning:
  enabled: true
reasoning:
  enabled: false
stage_args:
  captioning:
    - --image-prompt-file
    - ../prompts/captioning/image_prompt.md
    - --summary-prompt-file=../prompts/captioning/summary_prompt.md
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stage_arg == [
        "captioning=--image-prompt-file",
        f"captioning={image_prompt}",
        f"captioning=--summary-prompt-file={summary_prompt}",
    ]
    normalized = normalize_entries_for_container(entries)
    with _runner_input_file(normalized) as input_file:
        plan = build_container_plan(args, normalized, input_file=input_file)

    assert any(mount.source == prompt_dir and mount.read_only for mount in plan.mounts)


def test_repo_cookbook_paths_resolve_from_repo_root_when_cwd_is_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    (repo_root / ".git").mkdir()
    service_dir = repo_root / "services" / "workflow_runner"
    service_dir.mkdir(parents=True)
    media_dir = repo_root / "data"
    media_dir.mkdir()
    media = media_dir / "video.mp4"
    media.touch()
    cookbook_root = repo_root / "cookbooks" / "default_video"
    config_dir = cookbook_root / "configs"
    config_dir.mkdir(parents=True)
    question_bank = cookbook_root / "question_bank.json"
    question_bank.write_text('{"questions": []}\n', encoding="utf-8")
    config = config_dir / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
runtime:
  model_cache_path: ckpts
data:
  - inputs:
      media_path: data/video.mp4
    output:
      out_dir: output/auto_labeling/default_video
visual_qa:
  enabled: true
  question_bank_file: cookbooks/default_video/question_bank.json
reasoning:
  enabled: false
stage_args:
  captioning:
    - --window-seconds
    - 4
    - --window-frames
    - 0
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(service_dir)
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})

    captioning_args = _stage_flag_values(_stage_arg_values(args.stage_arg, "captioning"))
    assert entries[0].media_path == str(media)
    assert entries[0].data_path == str(repo_root / "output" / "auto_labeling" / "default_video")
    assert args.model_cache_path == str(repo_root / "ckpts")
    assert args.question_bank_file == str(question_bank)
    assert captioning_args["--window-seconds"] == "4"
    assert captioning_args["--window-frames"] == "0"


def test_interleaved_stage_arg_paths_are_tracked_per_stage(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    caption_prompt_dir = tmp_path / "caption_prompts"
    vqa_prompt_dir = tmp_path / "vqa_prompts"
    caption_prompt_dir.mkdir()
    vqa_prompt_dir.mkdir()
    caption_prompt = caption_prompt_dir / "caption.md"
    vqa_prompt = vqa_prompt_dir / "vqa.md"
    caption_prompt.write_text("Caption this clip.\n", encoding="utf-8")
    vqa_prompt.write_text("Ask visual questions.\n", encoding="utf-8")
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["captioning", "visual_qa"]
    args.stage_arg = [
        "captioning=--image-prompt-file",
        "visual_qa=--visual-qa-prompt-file",
        f"captioning={caption_prompt}",
        f"visual_qa={vqa_prompt}",
    ]

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert any(mount.source == caption_prompt_dir and mount.read_only for mount in plan.mounts)
    assert any(mount.source == vqa_prompt_dir and mount.read_only for mount in plan.mounts)


def test_cookbook_repo_root_style_missing_outputs_resolve_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    media_dir = tmp_path / "data"
    media_dir.mkdir()
    media = media_dir / "clip.mp4"
    media.touch()
    config_dir = tmp_path / "cookbooks" / "example" / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
runtime:
  model_cache_path: ckpts
data:
  - inputs:
      media_path: data/clip.mp4
    output:
      out_dir: output/auto_labeling/example
captioning:
  enabled: true
reasoning:
  enabled: false
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert entries[0].media_path == str(media)
    assert entries[0].data_path == str(tmp_path / "output" / "auto_labeling" / "example")
    assert args.model_cache_path == str(tmp_path / "ckpts")
    assert not (config_dir / "output").exists()


def test_cookbook_entries_reject_missing_media_or_output(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
data:
  - inputs:
      media_path: data/image.png
captioning:
  enabled: true
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    with pytest.raises(ValueError, match="Cookbook data entry 0"):
        apply_cookbook_config(args, protected_options={"--cookbook-file"})


def test_cookbook_config_supports_source_style_pipeline_settings(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        f"""
pipeline:
  model_cache_path: {tmp_path / "ckpts"}
  gpu_ids: 0
captioning:
  enabled: true
reasoning:
  enabled: true
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.pipeline == "image"
    assert args.model_cache_path == str(tmp_path / "ckpts")
    assert args.gpu_ids == "0"
    assert args.stages == ["captioning", "reasoning"]


def test_cookbook_stage_toggles_do_not_override_protected_stages(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
captioning:
  enabled: true
visual_qa:
  enabled: true
reasoning:
  enabled: true
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = ["captioning"]

    apply_cookbook_config(args, protected_options={"--cookbook-file", "--stages"})

    assert args.stages == ["captioning"]


def test_cookbook_stage_toggles_do_not_override_explicit_cookbook_stages(
    tmp_path: Path,
) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
stages:
  - captioning
captioning:
  enabled: true
visual_qa:
  enabled: true
reasoning:
  enabled: true
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stages == ["captioning"]


def test_cookbook_workflow_nodes_define_topological_stage_order(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
workflow:
  nodes:
    export:
      stage: reasoning
      needs: [qa]
    qa:
      stage: visual_qa
      needs: [captions]
    captions:
      stage: captioning
      needs: [tracking]
    tracking:
      stage: detection_and_tracking
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stages == ["detection_and_tracking", "captioning", "visual_qa", "reasoning"]


def test_cookbook_workflow_nodes_preserve_topological_order_for_independent_branches(
    tmp_path: Path,
) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
workflow:
  nodes:
    captions:
      stage: captioning
    tracking:
      stage: detection_and_tracking
    export:
      stage: reasoning
      needs: [captions, tracking]
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stages == ["captioning", "detection_and_tracking", "reasoning"]


def test_cookbook_workflow_nodes_reject_unknown_dependencies(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
workflow:
  nodes:
    captions:
      stage: captioning
      needs: [tracking]
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.stages = None

    with pytest.raises(ValueError, match="unknown node"):
        apply_cookbook_config(args, protected_options={"--cookbook-file"})


def test_cookbook_workflow_nodes_reject_dependency_cycles(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
workflow:
  nodes:
    captions:
      stage: captioning
      needs: [export]
    export:
      stage: reasoning
      needs: [captions]
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.stages = None

    with pytest.raises(ValueError, match="dependency cycle"):
        apply_cookbook_config(args, protected_options={"--cookbook-file"})


def test_cookbook_workflow_nodes_repeat_stage_with_node_args_and_mounts(tmp_path: Path) -> None:
    event_bank = tmp_path / "banks" / "event.json"
    person_bank = tmp_path / "banks" / "person.json"
    event_bank.parent.mkdir()
    event_bank.write_text('{"questions": []}\n', encoding="utf-8")
    person_bank.write_text('{"questions": []}\n', encoding="utf-8")
    media = tmp_path / "clip.mp4"
    media.touch()
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        f"""
pipeline: video
workflow:
  nodes:
    person_attributes:
      stage: visual_qa
      needs: [event_verification]
      args:
        - --question-bank-file
        - {person_bank}
        - --generation-mode
        - window-direct-vlm
        - --output-items-sidecar
        - visual_qa_person/items.json
    captions:
      stage: captioning
    event_verification:
      stage: visual_qa
      needs: [captions]
      args:
        - --question-bank-file={event_bank}
        - --generation-mode
        - window-vlm-llm
        - --output-items-sidecar
        - visual_qa_event/items.json
stage_args:
  visual_qa:
    - --generation-mode
    - metadata-llm
data:
  - inputs:
      media_path: {media}
    output:
      out_dir: {tmp_path / "scene"}
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})
    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)
        runner = ContainerPipelineRunner(
            config=ContainerRunnerConfig(mounts=plan.mounts),
            policy=EmptyOutputPolicy.FAIL,
            logger=logging.getLogger("test"),
        )
        completed: subprocess.CompletedProcess[list[str]] = subprocess.CompletedProcess(
            args=[], returncode=0
        )
        with patch("workflow_runner.container_runner.subprocess.run", return_value=completed):
            results = runner.run(list(plan.stages))

    assert [stage.node_id for stage in plan.stages] == [
        "captions",
        "event_verification",
        "person_attributes",
    ]
    assert [stage.name for stage in plan.stages] == ["captioning", "visual_qa", "visual_qa"]
    event_args = plan.stages[1].args
    person_args = plan.stages[2].args
    assert event_args[-5:] == (
        f"--question-bank-file={event_bank.resolve()}",
        "--generation-mode",
        "window-vlm-llm",
        "--output-items-sidecar",
        "visual_qa_event/items.json",
    )
    assert person_args[-6:] == (
        "--question-bank-file",
        str(person_bank.resolve()),
        "--generation-mode",
        "window-direct-vlm",
        "--output-items-sidecar",
        "visual_qa_person/items.json",
    )
    assert event_args.index("metadata-llm") < event_args.index("window-vlm-llm")
    assert person_args.index("metadata-llm") < person_args.index("window-direct-vlm")
    assert any(mount.source == event_bank.parent.resolve() for mount in plan.mounts)
    assert [result.node_id for result in results] == [
        "captions",
        "event_verification",
        "person_attributes",
    ]
    log_path = tmp_path / "scene" / "logs" / "workflow_runner.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["node_id"] for event in events] == [
        "captions",
        "captions",
        "event_verification",
        "event_verification",
        "person_attributes",
        "person_attributes",
    ]


def test_cookbook_workflow_nodes_reject_duplicate_node_ids(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
workflow:
  nodes:
    - id: qa
      stage: visual_qa
    - id: qa
      stage: visual_qa
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.stages = None

    with pytest.raises(ValueError, match="Duplicate workflow node id"):
        apply_cookbook_config(args, protected_options={"--cookbook-file"})


def test_cookbook_config_rejects_conflicting_stage_aliases(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
captioning:
  enabled: true
dense_caption:
  enabled: false
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.stages = None

    try:
        apply_cookbook_config(args, protected_options={"--cookbook-file"})
    except ValueError as exc:
        message = str(exc)
        assert "Conflicting enabled values for stage captioning" in message
        assert "captioning=True" in message
        assert "dense_caption=False" in message
    else:
        raise AssertionError("apply_cookbook_config should reject conflicting aliases")


def test_container_plan_includes_person_attribute_search_stage(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    pas_config = tmp_path / "pas.yaml"
    pas_config.write_text("attributes: {}\n", encoding="utf-8")
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["detection_and_tracking", "captioning", "person_attribute_search"]
    args.pas_config_file = str(pas_config)

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == [
        "detection_and_tracking",
        "captioning",
        "person_attribute_search",
    ]
    pas_stage = plan.stages[-1]
    assert pas_stage.image == "paidf-event-and-person-attribute-search-service"
    assert pas_stage.build_target == "event-and-person-attribute-search-service:build"
    assert "--config-file" in pas_stage.args
    assert str(pas_config.resolve()) in pas_stage.args
    # The service remains PAS-only, but PAS can use the shared LLM for query generation.
    assert "--vlm-endpoint-url" not in pas_stage.args
    assert pas_stage.args[pas_stage.args.index("--llm-endpoint-url") + 1] == args.llm_endpoint_url
    assert pas_stage.args[pas_stage.args.index("--llm-model") + 1] == args.llm_model
    assert "--reasoning-mode" not in pas_stage.args


def test_cookbook_node_pas_paths_are_resolved_and_mounted(tmp_path: Path) -> None:
    attributes = tmp_path / "inputs" / "attributes.json"
    prompt = tmp_path / "prompts" / "queries.json"
    image_group = tmp_path / "views"
    attributes.parent.mkdir()
    prompt.parent.mkdir()
    image_group.mkdir()
    attributes.write_text('{"attributes": {}}\n', encoding="utf-8")
    prompt.write_text('{"prompt": "query"}\n', encoding="utf-8")
    media = tmp_path / "representative.jpg"
    media.touch()
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
workflow:
  nodes:
    pas:
      stage: person_attribute_search
      args:
        - --attribute-json
        - inputs/attributes.json
        - --query-prompt-file=prompts/queries.json
        - --image-group-dir
        - views
data:
  - inputs:
      media_path: representative.jpg
    output:
      out_dir: output/pas
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    entries = apply_cookbook_config(args, protected_options={"--cookbook-file"})
    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    pas_args = plan.stages[0].args
    assert pas_args[-5:] == (
        "--attribute-json",
        str(attributes.resolve()),
        f"--query-prompt-file={prompt.resolve()}",
        "--image-group-dir",
        str(image_group.resolve()),
    )
    mounted_sources = {mount.source for mount in plan.mounts}
    assert attributes.parent.resolve() in mounted_sources
    assert prompt.parent.resolve() in mounted_sources
    assert image_group.resolve() in mounted_sources


def test_cookbook_forwards_config_to_person_attribute_search_stage(
    tmp_path: Path,
) -> None:
    cookbook = tmp_path / "pipeline_video.yaml"
    cookbook.write_text(
        "pipeline: video\n"
        "stages:\n"
        "  - detection_and_tracking\n"
        "  - visual_qa\n"
        "  - person_attribute_search\n"
        "data:\n"
        "  - inputs:\n"
        "      media_path: data/clip.mp4\n"
        "    output:\n"
        "      out_dir: output/pas\n"
        "person_attribute_search:\n"
        "  llm_query_generation: true\n",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(cookbook)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stages == [
        "detection_and_tracking",
        "visual_qa",
        "person_attribute_search",
    ]
    # The cookbook is forwarded to the PAS stage as --config-file so its
    # person_attribute_search block (e.g. llm_query_generation) reaches the
    # service.
    assert args.pas_config_file == str(cookbook)


def test_legacy_daft_export_stage_alias_resolves_to_reasoning(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    entries = normalize_entries_for_container(
        [DataEntry(media_path=str(media), data_path=str(tmp_path / "scene"))]
    )
    args = _runner_args()
    args.stages = ["captioning", "daft_export"]

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == ["captioning", "reasoning"]
    assert plan.stages[-1].image == "paidf-reasoning-service"


def test_cookbook_legacy_daft_export_section_maps_to_reasoning(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
captioning:
  enabled: true
daft_export:
  enabled: true
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.stages == ["captioning", "reasoning"]


def test_remote_paths_are_not_mounted_or_rewritten(tmp_path: Path) -> None:
    entry = DataEntry(media_path="msc://bucket/clip.mp4", data_path="msc://bucket/scene")
    entries = normalize_entries_for_container([entry])

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(_runner_args(), entries, input_file=input_file)

    assert entries[0].media_path == "msc://bucket/clip.mp4"
    assert entries[0].data_path == "msc://bucket/scene"
    assert all("msc://" not in str(mount.source) for mount in plan.mounts)


def test_stage_arg_requires_stage_prefix() -> None:
    try:
        parse_stage_args(("captioning",))
    except ValueError as exc:
        assert "expected STAGE=ARG" in str(exc)
    else:
        raise AssertionError("parse_stage_args should reject malformed pass-through args")


def test_main_dry_run_does_not_launch_containers(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    data_dir = tmp_path / "scene"
    payload = json.dumps([{"media_path": str(media), "data_path": str(data_dir)}])
    argv = [
        "workflow-runner",
        "--container-dry-run",
        "--input",
        payload,
    ]

    with (
        patch.object(sys, "argv", argv),
        patch("workflow_runner.container_runner.subprocess.run") as subprocess_run,
    ):
        main()

    subprocess_run.assert_not_called()


def test_execute_applies_dev_data_root_to_cookbook_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    original = [DataEntry(media_path=str(media), data_path=str(tmp_path / "source"))]
    copied = [DataEntry(media_path=str(media), data_path=str(tmp_path / "dev" / "entry"))]
    normalized: list[DataEntry] = []
    service = WorkflowRunnerService()
    args = _runner_args()
    args.dev_data_root = "dev"
    args.no_expand_input_dirs = True
    args.stages = ["captioning"]
    main_module = importlib.import_module("workflow_runner.main")

    monkeypatch.setattr(
        main_module,
        "apply_cookbook_config",
        lambda _args, **_kwargs: original,
    )
    monkeypatch.setattr(
        service,
        "_copy_data_entries_to_dev_root",
        lambda entries, dev_root: copied if entries == original and dev_root == "dev" else [],
    )

    def capture_normalize(data_entries: list[DataEntry]) -> list[DataEntry]:
        normalized.extend(data_entries)
        return data_entries

    monkeypatch.setattr(main_module, "normalize_entries_for_container", capture_normalize)

    service.execute(args, [])

    assert normalized == copied


def test_resolve_container_user_expands_auto_to_current_ids() -> None:
    assert resolve_container_user(None) is None
    assert resolve_container_user("1000:1000") == "1000:1000"
    assert resolve_container_user("none") == "none"
    assert resolve_container_user("auto") == f"{os.getuid()}:{os.getgid()}"


def test_rootless_env_defaults_injected_for_non_root_user() -> None:
    env = rootless_env_with_defaults("1004:1005", ())

    names = {item.split("=", 1)[0] for item in env}
    assert names == {"USER", "HOME", "HF_HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR"}
    assert "HOME=/tmp" in env


def test_rootless_env_defaults_skipped_for_root_or_unset_user() -> None:
    assert rootless_env_with_defaults(None, ("KEEP=1",)) == ("KEEP=1",)
    assert rootless_env_with_defaults("0:0", ()) == ()
    assert rootless_env_with_defaults("root", ()) == ()


def test_rootless_env_defaults_do_not_override_caller_values() -> None:
    env = rootless_env_with_defaults("1004:1005", ("HOME=/custom", "USER=alice"))

    assert "HOME=/custom" in env
    assert "USER=alice" in env
    assert "HOME=/tmp" not in env
    assert "HF_HOME=/tmp/hf" in env


def test_apply_container_defaults_resolves_user_and_injects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force a non-root uid/gid so the test is deterministic regardless of the
    # user running it. CI runs as root, where rootless env is intentionally
    # skipped; pinning the ids keeps the auto-resolution path under test.
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)
    args = WorkflowRunnerService().build_parser().parse_args([])
    args.container_user = "auto"

    _apply_container_defaults(args)

    assert args.container_user == "1000:1000"
    names = {item.split("=", 1)[0] for item in args.container_env}
    assert {"USER", "HOME", "HF_HOME"} <= names


def test_apply_container_defaults_leaves_root_runs_untouched() -> None:
    args = WorkflowRunnerService().build_parser().parse_args([])

    _apply_container_defaults(args)

    assert args.container_user is None
    assert args.container_env == []


def test_cookbook_container_section_sets_user_images_env_and_mounts(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
detection_and_tracking:
  enabled: true
person_attribute_search:
  enabled: true
container:
  user: auto
  images:
    detection_and_tracking: detection-and-tracking-sam3-service
  env:
    SAM3_MODEL_PATH: /models/sam3
  mounts:
    - /host/models/sam3:/models/sam3:ro
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.container_user == "auto"
    assert args.tracking_image == "detection-and-tracking-sam3-service"
    assert "SAM3_MODEL_PATH=/models/sam3" in args.container_env
    assert "/host/models/sam3:/models/sam3:ro" in args.container_mount


def test_cookbook_container_section_respects_protected_cli_flags(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
detection_and_tracking:
  enabled: true
container:
  user: auto
  images:
    detection_and_tracking: detection-and-tracking-sam3-service
  env:
    SAM3_MODEL_PATH: /models/sam3
  mounts:
    - /host/models/sam3:/models/sam3:ro
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None
    args.container_user = "2000:2000"
    args.tracking_image = "custom-tracking-image"
    args.container_env = ["EXISTING=1"]
    args.container_mount = ["/cli/mount:/cli/mount:rw"]

    apply_cookbook_config(
        args,
        protected_options={
            "--cookbook-file",
            "--container-user",
            "--tracking-image",
            "--container-env",
            "--container-mount",
        },
    )

    # Scalars (user, image) respect CLI protection; additive env/mounts merge
    # with cookbook values, and CLI entries are kept (placed last so they win).
    assert args.container_user == "2000:2000"
    assert args.tracking_image == "custom-tracking-image"
    assert args.container_env == ["SAM3_MODEL_PATH=/models/sam3", "EXISTING=1"]
    assert args.container_mount == [
        "/host/models/sam3:/models/sam3:ro",
        "/cli/mount:/cli/mount:rw",
    ]


def test_cookbook_training_export_section_sets_export_args(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
training_export:
  enabled: true
  formats:
    - cosmos-reason-v1.0
    - tao-vl-reason-v1.0
  output_dir: ./exports/training-section
  task_types: [mcq, open_qa]
  copy_media: false
  emit_media_root_as_null: true
  metadata:
    description: QA training split
    license: internal
    tags: [traffic, qa]
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.training_export_formats == ["cosmos-reason-v1.0", "tao-vl-reason-v1.0"]
    assert args.training_export_dir == str(tmp_path / "exports" / "training-section")
    assert args.training_export_tasks == ["mcq", "open_qa"]
    assert args.training_export_no_copy_media is True
    assert args.training_export_emit_media_root_as_null is True
    assert args.training_export_description == "QA training split"
    assert args.training_export_license == "internal"
    assert args.training_export_tags == ["traffic", "qa"]
    assert "training_export" in args.stages


def test_cookbook_training_export_output_dir_uses_shared_path_resolver(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
training_export:
  enabled: true
  output_dir: services/workflow_runner/exports
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.training_export_dir == str(tmp_path / "services" / "workflow_runner" / "exports")


def test_cookbook_training_export_parses_quoted_boolean_values(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
training_export:
  enabled: "true"
  formats: cosmos-reason-v1.0
  emit_media_root_as_null: "false"
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None
    args.training_export_emit_media_root_as_null = True

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.training_export_formats == ["cosmos-reason-v1.0"]
    assert args.training_export_emit_media_root_as_null is False


def test_cookbook_training_export_quoted_false_disables_section(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_video.yaml"
    config.write_text(
        """
pipeline: video
training_export:
  enabled: "false"
  formats: cosmos-reason-v1.0
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})

    assert args.training_export_formats == []


def test_container_plan_includes_grounding_2d_stage(tmp_path: Path) -> None:
    media = tmp_path / "scene.jpg"
    media.write_bytes(b"fake")
    entries = normalize_entries_for_container(
        [DataEntry(id="s1", media_path=str(media), data_path=str(tmp_path / "out"))]
    )
    args = _runner_args()
    args.pipeline = "image"
    args.stages = ["grounding_2d"]
    args.grounding_2d_image = "grounding-2d-service"
    args.model_cache_path = str(tmp_path / "ckpts")

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == ["grounding_2d"]
    grounding = plan.stages[0]
    assert grounding.image == "grounding-2d-service"
    assert "--vlm-endpoint-url" in grounding.args
    assert "--sam3-model-cache-path" in grounding.args


def test_container_plan_runs_captioning_before_grounding_2d(tmp_path: Path) -> None:
    media = tmp_path / "scene.jpg"
    media.write_bytes(b"fake")
    entries = normalize_entries_for_container(
        [DataEntry(id="s1", media_path=str(media), data_path=str(tmp_path / "out"))]
    )
    args = _runner_args()
    args.pipeline = "image"
    args.stages = ["grounding_2d", "captioning"]
    args.captioning_image = "captioning-service"
    args.grounding_2d_image = "grounding-2d-service"

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == ["captioning", "grounding_2d"]


def test_cookbook_selects_grounding_2d_stage(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
stages:
  - grounding_2d
container:
  images:
    grounding_2d: grounding-2d-service
endpoints:
  vlm:
    url: http://vlm:8000/v1
    model: test-vlm
grounding_2d:
  enabled: true
stage_args:
  grounding_2d:
    - --filter-ungroundable-expressions
    - --min-instance-score
    - "0.5"
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})
    _apply_container_defaults(args)

    assert args.stages == ["grounding_2d"]
    assert args.grounding_2d_image == "grounding-2d-service"
    assert args.vlm_endpoint_url == "http://vlm:8000/v1"
    assert "--filter-ungroundable-expressions" in _stage_arg_values(
        list(args.stage_arg), "grounding_2d"
    )


def test_container_plan_includes_referring_expressions_stage(tmp_path: Path) -> None:
    media = tmp_path / "scene.jpg"
    media.write_bytes(b"fake")
    entries = normalize_entries_for_container(
        [DataEntry(id="s1", media_path=str(media), data_path=str(tmp_path / "out"))]
    )
    args = _runner_args()
    args.pipeline = "image"
    args.stages = ["detection_and_tracking", "referring_expressions"]
    args.tracker = "sam3"
    args.referring_expressions_image = "referring-expressions-service"
    args.tracking_image = "detection-and-tracking-sam3-service"

    with _runner_input_file(entries) as input_file:
        plan = build_container_plan(args, entries, input_file=input_file)

    assert [stage.name for stage in plan.stages] == [
        "detection_and_tracking",
        "referring_expressions",
    ]
    referring = plan.stages[1]
    assert referring.image == "referring-expressions-service"
    assert "--vlm-endpoint-url" in referring.args


def test_cookbook_selects_referring_expressions_stages(tmp_path: Path) -> None:
    config = tmp_path / "pipeline_image.yaml"
    config.write_text(
        """
pipeline: image
stages:
  - detection_and_tracking
  - referring_expressions
container:
  images:
    detection_and_tracking: detection-and-tracking-sam3-service
    referring_expressions: referring-expressions-service
endpoints:
  vlm:
    url: http://vlm:8000/v1
    model: test-vlm
detection_and_tracking:
  enabled: true
  tracker: sam3
referring_expressions:
  enabled: true
stage_args:
  referring_expressions:
    - --draw-box-overlay
    - --min-match-iou
    - "0.3"
data: []
""",
        encoding="utf-8",
    )
    args = _runner_args()
    args.cookbook_file = str(config)
    args.pipeline = None
    args.stages = None

    apply_cookbook_config(args, protected_options={"--cookbook-file"})
    _apply_container_defaults(args)

    assert args.stages == ["detection_and_tracking", "referring_expressions"]
    assert args.referring_expressions_image == "referring-expressions-service"
    assert args.tracking_image == "detection-and-tracking-sam3-service"
    assert args.tracker == "sam3"
    assert args.vlm_endpoint_url == "http://vlm:8000/v1"
    assert "--draw-box-overlay" in _stage_arg_values(list(args.stage_arg), "referring_expressions")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
