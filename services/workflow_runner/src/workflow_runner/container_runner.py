# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container orchestration for Dockerized workflow stages."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from core import DataEntry, EmptyOutputPolicy, StagePolicyError, write_jsonl

RUNNER_MODULE_PATH = Path(__file__).resolve()

PipelineName = Literal["video", "image"]

StageName = Literal[
    "super_resolution",
    "detection_and_tracking",
    "referring_expressions",
    "captioning",
    "grounding_2d",
    "visual_qa",
    "reasoning",
    "training_export",
    "person_attribute_search",
]

STAGE_CHOICES: tuple[StageName, ...] = (
    "super_resolution",
    "detection_and_tracking",
    "referring_expressions",
    "captioning",
    "visual_qa",
    "reasoning",
    "training_export",
    "person_attribute_search",
)

# Selectable stages include image-only stages (e.g. grounding_2d) that are not
# part of the default video STAGE_CHOICES order.
SELECTABLE_STAGES: tuple[StageName, ...] = (
    "super_resolution",
    "detection_and_tracking",
    "referring_expressions",
    "captioning",
    "grounding_2d",
    "visual_qa",
    "reasoning",
    "training_export",
    "person_attribute_search",
)

# Legacy stage names accepted in CLI flags and cookbook configs, mapped to the
# current stage they were renamed to. ``daft_export`` was split on main into the
# ``reasoning`` service plus per-task DAFT validation, so old configs that name
# ``daft_export`` continue to resolve to the ``reasoning`` stage.
STAGE_ALIASES: dict[str, StageName] = {
    "daft_export": "reasoning",
    "tracking": "detection_and_tracking",
}

DEFAULT_STAGES: tuple[StageName, ...] = (
    "super_resolution",
    "detection_and_tracking",
    "captioning",
    "reasoning",
)

DEFAULT_PIPELINE: PipelineName = "video"

PIPELINE_STAGES: dict[PipelineName, tuple[StageName, ...]] = {
    "video": DEFAULT_STAGES,
    "image": (
        "captioning",
        "reasoning",
    ),
}

DEFAULT_STAGE_IMAGES: dict[StageName, str] = {
    "super_resolution": "paidf-super-resolution-service",
    "detection_and_tracking": "paidf-detection-and-tracking-rfdetr-service",
    "referring_expressions": "paidf-referring-expressions-service",
    "grounding_2d": "paidf-grounding-2d-service",
    "captioning": "paidf-captioning-service",
    "visual_qa": "paidf-visual-qa-service",
    "reasoning": "paidf-reasoning-service",
    "training_export": "paidf-training-export-service",
    "person_attribute_search": "paidf-event-and-person-attribute-search-service",
}

DEFAULT_BUILD_TARGETS: dict[StageName, str] = {
    "super_resolution": "super-resolution-service:build",
    "detection_and_tracking": "detection-and-tracking-service:rfdetr",
    "referring_expressions": "referring-expressions-service:main",
    "grounding_2d": "grounding-2d-service:main",
    "captioning": "captioning-service:main",
    "visual_qa": "visual-qa-service:build",
    "reasoning": "reasoning-service:build",
    "training_export": "training-export-service:build",
    "person_attribute_search": "event-and-person-attribute-search-service:build",
}

# Environment variables a rootless container needs when it runs as a non-root
# --user that has no entry in the image's /etc/passwd. Without them, libraries
# that call getpass.getuser() or write under $HOME/.cache crash. The runner
# injects these automatically whenever it runs as a non-root user so callers no
# longer have to pass them by hand.
ROOTLESS_ENV_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("USER", "appuser"),
    ("HOME", "/tmp"),
    ("HF_HOME", "/tmp/hf"),
    ("XDG_CACHE_HOME", "/tmp/.cache"),
    ("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor"),
)

# User values that denote root (or "no --user"); rootless defaults do not apply.
_ROOT_USER_VALUES: frozenset[str] = frozenset({"", "none", "0", "0:0", "root", "root:root"})

ALLOWED_RUNTIMES: frozenset[str] = frozenset({"docker", "podman"})
REMOTE_URI_MARKERS: tuple[str, ...] = ("://",)
PATH_STAGE_ARG_FLAGS: frozenset[str] = frozenset(
    {
        "--attribute-json",
        "--config-file",
        "--image-group-dir",
        "--image-prompt-file",
        "--prompt-file",
        "--question-bank-file",
        "--query-prompt-file",
        "--summary-prompt-file",
        "--evidence-prompt-file",
        "--visual-qa-evidence-prompt-file",
        "--visual-qa-prompt-file",
        "--visual-qa-question-bank-file",
    }
)
RESERVED_STAGE_ARG_FLAGS: frozenset[str] = frozenset({"--input-file", "--log-level"})


@dataclass(frozen=True)
class WorkflowNode:
    """One declarative node in an ordered workflow execution plan."""

    node_id: str
    stage: StageName
    needs: tuple[str, ...] = ()
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class VolumeMount:
    """One bind mount passed to the container runtime."""

    source: Path
    target: str
    read_only: bool = False

    def as_mount_arg(self) -> str:
        """Render this mount in Docker/Podman ``--mount`` syntax."""
        parts = [
            "type=bind",
            f"source={self.source}",
            f"target={self.target}",
        ]
        if self.read_only:
            parts.append("readonly")
        return ",".join(parts)


@dataclass(frozen=True)
class ContainerStage:
    """A pipeline stage executed as one container invocation."""

    name: StageName
    image: str
    args: tuple[str, ...]
    build_target: str
    log_paths: tuple[Path, ...] = ()
    node_id: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            object.__setattr__(self, "node_id", self.name)


@dataclass(frozen=True)
class StageRunResult:
    """Result from one container stage execution."""

    stage: StageName
    command: tuple[str, ...]
    returncode: int
    skipped: bool = False
    node_id: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            object.__setattr__(self, "node_id", self.stage)

    @property
    def success(self) -> bool:
        """Whether the stage completed successfully."""
        return self.returncode == 0


@dataclass(frozen=True)
class ContainerRunnerConfig:
    """Runtime settings shared by all stage containers."""

    runtime: str = "docker"
    gpus: str | None = "all"
    network: str | None = "host"
    remove: bool = True
    workdir: str | None = None
    user: str | None = None
    env: tuple[str, ...] = ()
    mounts: tuple[VolumeMount, ...] = ()
    dry_run: bool = False
    build_images: bool = False
    ensure_images: bool = False


@dataclass(frozen=True)
class ContainerPipelinePlan:
    """Complete container pipeline plan."""

    stages: tuple[ContainerStage, ...]
    mounts: tuple[VolumeMount, ...]
    input_file: Path


class ContainerPipelineRunner:
    """Run workflow stages as separate containers."""

    def __init__(
        self,
        *,
        config: ContainerRunnerConfig,
        policy: EmptyOutputPolicy,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.policy = policy
        self.logger = logger

    def run(self, stages: list[ContainerStage]) -> list[StageRunResult]:
        """Execute each stage in order and return stage results."""
        if self.config.build_images or self.config.ensure_images:
            if not self.config.dry_run:
                _validate_build_script_available()
            force = self.config.build_images
            for stage in stages:
                # --container-build-images forces a rebuild of every stage.
                # --container-ensure-images only builds images that are absent;
                # in dry-run we cannot probe the daemon so we surface the build
                # command for each stage as if it were missing.
                if force or self.config.dry_run or not self._image_exists(stage.image):
                    self._build_stage_image(stage)
                else:
                    self.logger.info(
                        "Image %s for node %s (stage %s) already present; skipping build.",
                        stage.image,
                        _stage_node_id(stage),
                        stage.name,
                    )

        results: list[StageRunResult] = []
        for stage in stages:
            command = tuple(self.stage_command(stage))
            self.logger.info(
                "Running container node %s (stage %s): %s",
                _stage_node_id(stage),
                stage.name,
                _quote(command),
            )
            if self.config.dry_run:
                results.append(
                    StageRunResult(
                        stage=stage.name,
                        command=command,
                        returncode=0,
                        skipped=True,
                        node_id=_stage_node_id(stage),
                    )
                )
                continue

            self._write_stage_event(
                stage,
                {
                    "time": _utc_timestamp(),
                    "node_id": _stage_node_id(stage),
                    "stage": stage.name,
                    "event": "start",
                    "image": stage.image,
                    "command": list(command),
                },
            )
            completed = subprocess.run(command, check=False)
            result = StageRunResult(
                stage=stage.name,
                command=command,
                returncode=completed.returncode,
                node_id=_stage_node_id(stage),
            )
            results.append(result)
            self._write_stage_event(
                stage,
                {
                    "time": _utc_timestamp(),
                    "node_id": _stage_node_id(stage),
                    "stage": stage.name,
                    "event": "success" if result.success else "failure",
                    "returncode": result.returncode,
                },
            )
            if result.success:
                continue

            message = (
                f"Container stage {stage.name} failed (node {_stage_node_id(stage)}) "
                f"with exit code {result.returncode}"
            )
            if self.policy is EmptyOutputPolicy.FAIL:
                raise StagePolicyError(message)
            self.logger.warning("%s; continuing because policy=warn.", message)

        return results

    def _write_stage_event(self, stage: ContainerStage, event: dict[str, object]) -> None:
        for log_path in stage.log_paths:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{json.dumps(event, sort_keys=True)}\n")
            except OSError as exc:
                self.logger.warning("Could not write workflow runner log %s: %s", log_path, exc)

    def stage_command(self, stage: ContainerStage) -> list[str]:
        """Build the container runtime command for one stage."""
        runtime = _validate_runtime(self.config.runtime)
        command = [runtime, "run"]
        if self.config.remove:
            command.append("--rm")
        if self.config.gpus:
            command.extend(["--gpus", self.config.gpus])
        if self.config.network:
            command.extend(["--network", self.config.network])
        if self.config.workdir:
            command.extend(["--workdir", self.config.workdir])
        if self.config.user:
            command.extend(["--user", self.config.user])
        for env_name in self.config.env:
            command.extend(["--env", env_name])
        for mount in self.config.mounts:
            command.extend(["--mount", mount.as_mount_arg()])
        command.append(stage.image)
        command.extend(stage.args)
        return command

    def _image_exists(self, image: str) -> bool:
        """Return True when the container image is already present locally."""
        runtime = _validate_runtime(self.config.runtime)
        completed = subprocess.run(
            [runtime, "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0

    def _build_stage_image(self, stage: ContainerStage) -> None:
        build_script = _build_script_path()
        command = [
            "uv",
            "run",
            "--group",
            "dev",
            "python",
            str(build_script),
            stage.build_target,
        ]
        self.logger.info(
            "Building image for node %s (stage %s) with target %s: %s",
            _stage_node_id(stage),
            stage.name,
            stage.build_target,
            _quote(tuple(command)),
        )
        if self.config.dry_run:
            return
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise StagePolicyError(
                f"Image build for node {_stage_node_id(stage)} (stage {stage.name}) failed "
                f"with exit code {completed.returncode}"
            )


def normalize_entries_for_container(data_entries: list[DataEntry]) -> list[DataEntry]:
    """Resolve local media/data paths so identity bind mounts work in containers."""
    normalized: list[DataEntry] = []
    for entry in data_entries:
        updates: dict[str, str] = {}
        if not _is_remote_uri(entry.media_path):
            updates["media_path"] = str(Path(entry.media_path).expanduser().resolve())
        if not _is_remote_uri(entry.data_path):
            data_path = Path(entry.data_path).expanduser().resolve()
            data_path.mkdir(parents=True, exist_ok=True)
            updates["data_path"] = str(data_path)
        normalized.append(entry.model_copy(update=updates))
    return normalized


def write_runner_input_file(
    data_entries: list[DataEntry],
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Write the JSONL manifest consumed by every stage container."""
    tempdir = tempfile.TemporaryDirectory(prefix="workflow_runner_")
    root = Path(tempdir.name)
    input_file = root / "input.jsonl"
    write_jsonl(input_file, (entry.model_dump(mode="json") for entry in data_entries))
    root.chmod(0o755)
    input_file.chmod(0o644)
    return input_file, tempdir


def build_container_plan(
    args: Namespace,
    data_entries: list[DataEntry],
    *,
    input_file: Path,
) -> ContainerPipelinePlan:
    """Build the ordered container stage plan from parsed service args."""
    workflow_nodes = tuple(getattr(args, "workflow_nodes", ()) or ())
    if workflow_nodes:
        selected_nodes = workflow_nodes
    else:
        selected = {_normalize_stage_name(stage) for stage in args.stages}
        selected_nodes = tuple(
            WorkflowNode(node_id=stage_name, stage=stage_name)
            for stage_name in _canonical_stage_order(_pipeline_from_args(args))
            if stage_name in selected
        )
    extra_mounts = tuple(parse_volume_mount(raw_mount) for raw_mount in args.container_mount)
    stage_extra_args = parse_stage_args(tuple(args.stage_arg))
    mounts = local_mounts_for_entries(
        data_entries,
        input_file=input_file,
        config_paths=_config_paths_from_args(args),
        cache_paths=_cache_paths_from_args(args),
        extra_mounts=extra_mounts,
    )
    log_paths = _workflow_log_paths(data_entries)

    stages = tuple(
        _stage_from_args(
            stage_name=node.stage,
            node_id=node.node_id,
            args=args,
            input_file=input_file,
            extra_args=(*stage_extra_args.get(node.stage, ()), *node.args),
            log_paths=log_paths,
        )
        for node in selected_nodes
    )
    return ContainerPipelinePlan(stages=stages, mounts=mounts, input_file=input_file)


def _canonical_stage_order(pipeline: PipelineName) -> tuple[StageName, ...]:
    if pipeline == "image":
        return (
            "detection_and_tracking",
            "referring_expressions",
            "captioning",
            "grounding_2d",
            "visual_qa",
            "reasoning",
            "training_export",
            "person_attribute_search",
        )
    return STAGE_CHOICES


def _pipeline_from_args(args: Namespace) -> PipelineName:
    value = getattr(args, "pipeline", None)
    if value == "image":
        return "image"
    return DEFAULT_PIPELINE


def local_mounts_for_entries(
    data_entries: list[DataEntry],
    *,
    input_file: Path,
    config_paths: tuple[str, ...] = (),
    cache_paths: tuple[str, ...] = (),
    extra_mounts: tuple[VolumeMount, ...] = (),
) -> tuple[VolumeMount, ...]:
    """Collect identity bind mounts needed by stage containers."""
    mounts: list[VolumeMount] = [
        VolumeMount(source=input_file.parent, target=str(input_file.parent), read_only=True)
    ]
    for config_path in config_paths:
        mounts.extend(_file_parent_mount(config_path))
    for cache_path in cache_paths:
        if _is_remote_uri(cache_path):
            continue
        path = Path(cache_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        mounts.append(VolumeMount(source=path, target=str(path)))
    for entry in data_entries:
        mounts.extend(_entry_mounts(entry))
    mounts.extend(extra_mounts)
    return _dedupe_mounts(mounts)


def resolve_container_user(value: object) -> str | None:
    """Resolve the ``--container-user`` value, expanding the ``auto`` sentinel.

    ``auto`` resolves to the current ``uid:gid`` so stage output files are owned
    by the invoking user instead of root. On platforms without ``os.getuid``
    (non-POSIX) the sentinel resolves to ``None`` so ``--user`` is omitted.
    """
    if value is None:
        return None
    rendered = str(value).strip()
    if rendered.lower() != "auto":
        return rendered
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return None
    return f"{getuid()}:{getgid()}"


def rootless_env_with_defaults(user: str | None, env: tuple[str, ...]) -> tuple[str, ...]:
    """Augment container env with rootless hygiene defaults for non-root users.

    Defaults are added only when running as an explicit non-root ``user`` and
    never override values the caller already supplied (matched by name).
    """
    if user is None or user.strip().lower() in _ROOT_USER_VALUES:
        return tuple(env)
    present = {item.split("=", 1)[0] for item in env}
    additions = tuple(
        f"{name}={value}" for name, value in ROOTLESS_ENV_DEFAULTS if name not in present
    )
    return (*env, *additions)


def runner_config_from_args(
    args: Namespace,
    mounts: tuple[VolumeMount, ...],
) -> ContainerRunnerConfig:
    """Build container runtime config from parsed service args."""
    gpus = _optional_runtime_value(args.container_gpus, disabled_values={"", "none", "false", "0"})
    network = _optional_runtime_value(args.container_network, disabled_values={"", "none"})
    user = _optional_runtime_value(args.container_user, disabled_values={"", "none"})
    return ContainerRunnerConfig(
        runtime=_validate_runtime(str(args.container_runtime)),
        gpus=gpus,
        network=network,
        workdir=args.container_workdir,
        user=user,
        env=tuple(args.container_env),
        mounts=mounts,
        dry_run=bool(args.container_dry_run),
        build_images=bool(args.container_build_images),
        ensure_images=bool(getattr(args, "container_ensure_images", False)),
    )


def parse_volume_mount(raw_mount: str) -> VolumeMount:
    """Parse ``HOST[:CONTAINER[:ro|rw]]`` into a volume mount."""
    parts = raw_mount.split(":")
    if len(parts) > 3:
        msg = f"Invalid mount {raw_mount!r}; expected HOST[:CONTAINER[:ro|rw]]."
        raise ValueError(msg)
    if not parts[0].strip():
        msg = f"Invalid mount {raw_mount!r}; host path must not be empty."
        raise ValueError(msg)
    source = Path(parts[0]).expanduser().resolve()
    if len(parts) >= 2 and parts[1]:
        target = parts[1]
        if not target.startswith("/"):
            msg = f"Invalid mount {raw_mount!r}; explicit container target must be absolute."
            raise ValueError(msg)
    else:
        target = str(source)
    mode = parts[2] if len(parts) == 3 else "rw"
    if mode not in {"ro", "rw"}:
        msg = f"Invalid mount mode in {raw_mount!r}; expected ro or rw."
        raise ValueError(msg)
    return VolumeMount(source=source, target=target, read_only=mode == "ro")


def parse_stage_args(raw_args: tuple[str, ...]) -> dict[StageName, tuple[str, ...]]:
    """Parse repeated ``STAGE=ARG`` pass-through arguments."""
    parsed: dict[StageName, list[str]] = {}
    for raw_arg in raw_args:
        stage_raw, separator, value = raw_arg.partition("=")
        if not separator or not value:
            msg = f"Invalid stage arg {raw_arg!r}; expected STAGE=ARG."
            raise ValueError(msg)
        _reject_reserved_stage_args((value,), source=raw_arg)
        stage = _normalize_stage_name(stage_raw)
        parsed.setdefault(stage, []).append(value)
    return {stage: tuple(values) for stage, values in parsed.items()}


def default_tracking_image(tracker: str) -> str:
    """Return the default tracking service image for a tracker backend."""
    return (
        "paidf-detection-and-tracking-sam3-service"
        if tracker.startswith("sam3")
        else DEFAULT_STAGE_IMAGES["detection_and_tracking"]
    )


def default_tracking_build_target(tracker: str) -> str:
    """Return the default build target for a tracker backend."""
    return (
        "detection-and-tracking-service:sam3"
        if tracker.startswith("sam3")
        else DEFAULT_BUILD_TARGETS["detection_and_tracking"]
    )


def default_stages_for_pipeline(pipeline: PipelineName) -> tuple[StageName, ...]:
    """Return the default stage sequence for a media pipeline."""
    return PIPELINE_STAGES[pipeline]


def _stage_from_args(
    *,
    stage_name: StageName,
    node_id: str,
    args: Namespace,
    input_file: Path,
    extra_args: tuple[str, ...],
    log_paths: tuple[Path, ...],
) -> ContainerStage:
    base_args = ("--log-level", str(args.log_level), "--input-file", str(input_file))
    image = _stage_image(args, stage_name)
    build_target = _stage_build_target(args, stage_name)
    stage_args = _stage_service_args(stage_name, args)
    _reject_reserved_stage_args(stage_args, source=f"{stage_name} service args")
    _reject_reserved_stage_args(extra_args, source=f"{stage_name} extra args")
    return ContainerStage(
        name=stage_name,
        image=image,
        build_target=build_target,
        args=(*base_args, *stage_args, *extra_args),
        log_paths=log_paths,
        node_id=node_id,
    )


def _stage_service_args(stage_name: StageName, args: Namespace) -> tuple[str, ...]:
    model_cache_path = _host_path_arg(args.model_cache_path)
    question_bank_file = _host_path_arg(args.question_bank_file)
    reasoning_config_file = _host_path_arg(args.reasoning_config_file)
    pas_config_file = _host_path_arg(getattr(args, "pas_config_file", None))
    if stage_name == "super_resolution":
        return (
            "--resolver",
            str(args.sr_resolver),
            *_optional_arg("--variant", args.sr_variant),
            *_optional_arg("--model-cache-path", model_cache_path),
            *_optional_arg("--gpu-ids", args.gpu_ids),
        )
    if stage_name == "detection_and_tracking":
        return (
            "--tracker",
            str(args.tracker),
            *_optional_arg("--model-cache-path", model_cache_path),
            *_optional_arg("--gpu-ids", args.gpu_ids),
            *_list_arg("--classes", tuple(args.classes)),
        )
    if stage_name == "referring_expressions":
        return (
            *_optional_arg("--vlm-endpoint-url", args.vlm_endpoint_url),
            *_optional_arg("--vlm-model", args.vlm_model),
        )
    if stage_name == "grounding_2d":
        return (
            *_optional_arg("--vlm-endpoint-url", args.vlm_endpoint_url),
            *_optional_arg("--vlm-model", args.vlm_model),
            *_optional_arg("--sam3-model-cache-path", model_cache_path),
            *_optional_arg("--sam3-gpu-ids", args.gpu_ids),
        )
    if stage_name == "captioning":
        return (
            *_optional_arg("--vlm-endpoint-url", args.vlm_endpoint_url),
            *_optional_arg("--vlm-model", args.vlm_model),
            *_optional_arg("--llm-endpoint-url", args.llm_endpoint_url),
            *_optional_arg("--llm-model", args.llm_model),
        )
    if stage_name == "visual_qa":
        return (
            *_optional_arg("--question-bank-file", question_bank_file),
            *_optional_arg("--vlm-endpoint-url", args.vlm_endpoint_url),
            *_optional_arg("--vlm-model", args.vlm_model),
            *_optional_arg("--llm-endpoint-url", args.llm_endpoint_url),
            *_optional_arg("--llm-model", args.llm_model),
        )
    if stage_name == "person_attribute_search":
        return (
            *_optional_arg("--config-file", pas_config_file),
            *_optional_arg("--llm-endpoint-url", args.llm_endpoint_url),
            *_optional_arg("--llm-model", args.llm_model),
        )
    if stage_name == "reasoning":
        return (
            *_optional_arg("--config-file", reasoning_config_file),
            *_optional_arg("--llm-endpoint-url", args.llm_endpoint_url),
            *_optional_arg("--llm-model", args.llm_model),
            "--reasoning-mode",
            str(args.reasoning_mode),
        )
    return _training_export_args(args)


def _stage_image(args: Namespace, stage_name: StageName) -> str:
    if stage_name == "super_resolution":
        return str(args.sr_image)
    if stage_name == "detection_and_tracking":
        return str(args.tracking_image)
    if stage_name == "referring_expressions":
        return str(args.referring_expressions_image)
    if stage_name == "grounding_2d":
        return str(args.grounding_2d_image)
    if stage_name == "captioning":
        return str(args.captioning_image)
    if stage_name == "visual_qa":
        return str(args.visual_qa_image)
    if stage_name == "training_export":
        return str(args.training_export_image)
    if stage_name == "person_attribute_search":
        return str(args.pas_image)
    return str(args.reasoning_image)


def _stage_build_target(args: Namespace, stage_name: StageName) -> str:
    if stage_name == "super_resolution":
        return str(args.sr_build_target)
    if stage_name == "detection_and_tracking":
        return str(args.tracking_build_target)
    if stage_name == "referring_expressions":
        return str(args.referring_expressions_build_target)
    if stage_name == "grounding_2d":
        return str(args.grounding_2d_build_target)
    if stage_name == "captioning":
        return str(args.captioning_build_target)
    if stage_name == "visual_qa":
        return str(args.visual_qa_build_target)
    if stage_name == "training_export":
        return str(args.training_export_build_target)
    if stage_name == "person_attribute_search":
        return str(args.pas_build_target)
    return str(args.reasoning_build_target)


def _entry_mounts(entry: DataEntry) -> list[VolumeMount]:
    mounts: list[VolumeMount] = []
    if not _is_remote_uri(entry.media_path):
        media = Path(entry.media_path).expanduser().resolve()
        mounts.append(VolumeMount(source=media.parent, target=str(media.parent), read_only=True))
    if not _is_remote_uri(entry.data_path):
        data_root = Path(entry.data_path).expanduser().resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        mounts.append(VolumeMount(source=data_root, target=str(data_root)))
    return mounts


def _workflow_log_paths(data_entries: list[DataEntry]) -> tuple[Path, ...]:
    deduped: dict[Path, None] = {}
    for entry in data_entries:
        if _is_remote_uri(entry.data_path):
            continue
        path = Path(entry.data_path).expanduser().resolve() / "logs" / "workflow_runner.jsonl"
        deduped.setdefault(path, None)
    return tuple(deduped)


def _file_parent_mount(path_value: str) -> tuple[VolumeMount, ...]:
    if _is_remote_uri(path_value):
        return ()
    path = Path(path_value).expanduser().resolve()
    if path.exists() and path.is_dir():
        return (VolumeMount(source=path, target=str(path), read_only=True),)
    return (VolumeMount(source=path.parent, target=str(path.parent), read_only=True),)


def _config_paths_from_args(args: Namespace) -> tuple[str, ...]:
    paths = [
        getattr(args, "cookbook_root", None),
        args.reasoning_config_file,
        getattr(args, "pas_config_file", None),
        args.question_bank_file,
        *_stage_arg_paths(tuple(args.stage_arg)),
        *_workflow_node_arg_paths(args),
    ]
    return tuple(str(path) for path in paths if path)


def _cache_paths_from_args(args: Namespace) -> tuple[str, ...]:
    paths = [
        args.model_cache_path,
        getattr(args, "training_export_dir", None)
        if getattr(args, "training_export_formats", None)
        and _stage_selected(args, "training_export")
        else None,
    ]
    return tuple(str(path) for path in paths if path)


def _host_path_arg(value: object) -> str | None:
    rendered = str(value) if value is not None else ""
    if not rendered:
        return None
    if _is_remote_uri(rendered):
        return rendered
    return str(Path(rendered).expanduser().resolve())


def _dedupe_mounts(mounts: list[VolumeMount]) -> tuple[VolumeMount, ...]:
    deduped: dict[tuple[Path, str], VolumeMount] = {}
    for mount in mounts:
        key = (mount.source, mount.target)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = mount
        elif existing.read_only and not mount.read_only:
            deduped[key] = mount
    return tuple(deduped.values())


def _optional_arg(flag: str, value: object) -> tuple[str, str] | tuple[()]:
    if value is None:
        return ()
    rendered = str(value)
    if not rendered:
        return ()
    return (flag, rendered)


def _list_arg(flag: str, values: tuple[object, ...]) -> tuple[str, ...]:
    rendered = tuple(str(value) for value in values if str(value))
    if not rendered:
        return ()
    return (flag, *rendered)


def _repeat_arg(flag: str, values: tuple[object, ...]) -> tuple[str, ...]:
    rendered = tuple(str(value) for value in values if str(value))
    return tuple(item for value in rendered for item in (flag, value))


def _training_export_args(args: Namespace) -> tuple[str, ...]:
    output_dir = _host_path_arg(getattr(args, "training_export_dir", None))
    return (
        *_repeat_arg(
            "--training-export-format",
            tuple(getattr(args, "training_export_formats", ()) or ()),
        ),
        *_optional_arg("--training-export-dir", output_dir),
        *_repeat_arg(
            "--training-export-task",
            tuple(getattr(args, "training_export_tasks", ()) or ()),
        ),
        *_optional_arg(
            "--training-export-description",
            getattr(args, "training_export_description", None),
        ),
        *_optional_arg("--training-export-license", getattr(args, "training_export_license", None)),
        *_repeat_arg(
            "--training-export-tag",
            tuple(getattr(args, "training_export_tags", ()) or ()),
        ),
        *(
            ("--training-export-no-copy-media",)
            if getattr(args, "training_export_no_copy_media", False)
            else ()
        ),
        *(
            ("--training-export-emit-media-root-as-null",)
            if getattr(args, "training_export_emit_media_root_as_null", False)
            else ()
        ),
    )


def _stage_selected(args: Namespace, stage_name: StageName) -> bool:
    stages = getattr(args, "stages", ()) or ()
    return stage_name in {_normalize_stage_name(str(stage)) for stage in stages}


def _stage_arg_paths(raw_args: tuple[str, ...]) -> tuple[str, ...]:
    paths: list[str] = []
    expect_path_by_stage: dict[StageName, bool] = {}
    for raw_arg in raw_args:
        stage_raw, separator, value = raw_arg.partition("=")
        if not separator:
            continue
        stage = _normalize_stage_name(stage_raw)
        if expect_path_by_stage.get(stage):
            paths.append(value)
            expect_path_by_stage.pop(stage, None)
            continue
        inline_path = _inline_path_stage_arg(value)
        if inline_path is not None:
            paths.append(inline_path)
            continue
        if value in PATH_STAGE_ARG_FLAGS:
            expect_path_by_stage[stage] = True
    return tuple(paths)


def _workflow_node_arg_paths(args: Namespace) -> tuple[str, ...]:
    paths: list[str] = []
    for node in tuple(getattr(args, "workflow_nodes", ()) or ()):
        tagged_args = tuple(f"{node.stage}={value}" for value in node.args)
        paths.extend(_stage_arg_paths(tagged_args))
    return tuple(paths)


def _inline_path_stage_arg(value: str) -> str | None:
    for flag in PATH_STAGE_ARG_FLAGS:
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _normalize_stage_name(stage: str) -> StageName:
    normalized = stage.strip().replace("-", "_")
    normalized = STAGE_ALIASES.get(normalized, normalized)
    for choice in SELECTABLE_STAGES:
        if normalized == choice:
            return choice
    choices = ", ".join(SELECTABLE_STAGES)
    msg = f"Invalid stage {stage!r}; expected one of: {choices}."
    raise ValueError(msg)


def _optional_runtime_value(value: object, *, disabled_values: set[str]) -> str | None:
    rendered = str(value).strip()
    if rendered.lower() in disabled_values:
        return None
    return rendered


def _validate_runtime(runtime: str) -> str:
    rendered = runtime.strip()
    if rendered not in ALLOWED_RUNTIMES:
        choices = ", ".join(sorted(ALLOWED_RUNTIMES))
        msg = f"Unsupported container runtime {runtime!r}; expected one of: {choices}."
        raise ValueError(msg)
    return rendered


def _is_remote_uri(value: str) -> bool:
    return any(marker in value for marker in REMOTE_URI_MARKERS)


def _build_script_path() -> Path:
    for parent in RUNNER_MODULE_PATH.parents:
        candidate = parent / "scripts" / "build.py"
        if candidate.exists():
            return candidate
    return RUNNER_MODULE_PATH.parents[4] / "scripts" / "build.py"


def _validate_build_script_available() -> None:
    build_script = _build_script_path()
    if not build_script.exists():
        raise StagePolicyError(
            "--container-build-images requires a source checkout with scripts/build.py; "
            f"build script not found at {build_script}"
        )


def _reject_reserved_stage_args(values: tuple[str, ...], *, source: str) -> None:
    for value in values:
        for flag in RESERVED_STAGE_ARG_FLAGS:
            if value == flag or value.startswith(f"{flag}="):
                raise ValueError(
                    f"Stage pass-through args must not set runner-owned flag {flag}; "
                    f"found {value!r} in {source}."
                )


def _quote(command: tuple[str, ...]) -> str:
    return " ".join(command)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stage_node_id(stage: ContainerStage) -> str:
    return stage.node_id or stage.name


__all__ = [
    "ALLOWED_RUNTIMES",
    "ContainerPipelinePlan",
    "ContainerPipelineRunner",
    "ContainerRunnerConfig",
    "ContainerStage",
    "DEFAULT_BUILD_TARGETS",
    "DEFAULT_PIPELINE",
    "DEFAULT_STAGE_IMAGES",
    "DEFAULT_STAGES",
    "PATH_STAGE_ARG_FLAGS",
    "PIPELINE_STAGES",
    "PipelineName",
    "RESERVED_STAGE_ARG_FLAGS",
    "ROOTLESS_ENV_DEFAULTS",
    "STAGE_ALIASES",
    "STAGE_CHOICES",
    "SELECTABLE_STAGES",
    "StageName",
    "StageRunResult",
    "VolumeMount",
    "WorkflowNode",
    "build_container_plan",
    "default_stages_for_pipeline",
    "default_tracking_build_target",
    "default_tracking_image",
    "local_mounts_for_entries",
    "normalize_entries_for_container",
    "parse_stage_args",
    "parse_volume_mount",
    "resolve_container_user",
    "rootless_env_with_defaults",
    "runner_config_from_args",
    "write_runner_input_file",
]
