# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cookbook configuration support for the workflow runner."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import yaml
from core import DataEntry

from workflow_runner.container_runner import (
    DEFAULT_PIPELINE,
    PATH_STAGE_ARG_FLAGS,
    SELECTABLE_STAGES,
    STAGE_ALIASES,
    PipelineName,
    StageName,
    WorkflowNode,
    default_stages_for_pipeline,
)

# Maps a cookbook ``container.images`` stage key to the runner arg + CLI flag
# that overrides that stage's image, so a cookbook can pin per-stage images.
_IMAGE_ARG_BY_STAGE: dict[StageName, tuple[str, str]] = {
    "super_resolution": ("sr_image", "--sr-image"),
    "detection_and_tracking": ("tracking_image", "--tracking-image"),
    "referring_expressions": ("referring_expressions_image", "--referring-expressions-image"),
    "grounding_2d": ("grounding_2d_image", "--grounding-2d-image"),
    "captioning": ("captioning_image", "--captioning-image"),
    "visual_qa": ("visual_qa_image", "--visual-qa-image"),
    "reasoning": ("reasoning_image", "--reasoning-image"),
    "training_export": ("training_export_image", "--training-export-image"),
    "person_attribute_search": ("pas_image", "--pas-image"),
}

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
)


def apply_cookbook_config(
    args: Namespace,
    *,
    protected_options: set[str] | None = None,
) -> list[DataEntry]:
    """Apply an optional cookbook config to runner args and return configured entries."""
    cookbook_file = getattr(args, "cookbook_file", None)
    if not cookbook_file:
        args.workflow_nodes = None
        _finalize_pipeline(args, explicit_stage_states={})
        _add_training_export_stage_if_configured(args, protected_options or explicit_cli_options())
        return []

    path = Path(str(cookbook_file)).expanduser().resolve()
    payload = _load_mapping(path)
    protected = set(protected_options) if protected_options is not None else explicit_cli_options()
    args.cookbook_root = str(path.parent.parent if path.parent.name == "configs" else path.parent)

    pipeline = _pipeline_from_payload(payload, path)
    if pipeline is not None and "--pipeline" not in protected:
        args.pipeline = pipeline

    stage_states = _stage_enabled_map(payload)
    workflow_nodes = _workflow_node_list(payload, config_dir=path.parent)
    workflow_stages = (
        tuple(node.stage for node in workflow_nodes) if workflow_nodes is not None else None
    )
    explicit_stages = (
        workflow_stages if workflow_stages is not None else _stage_list(payload.get("stages"))
    )
    if "--stages" not in protected and explicit_stages is not None:
        args.stages = list(explicit_stages)
        args.workflow_nodes = list(workflow_nodes) if workflow_nodes is not None else None
    elif "--stages" in protected:
        args.workflow_nodes = None

    _apply_runtime_section(args, payload, path.parent, protected)
    _apply_container_section(args, payload, protected)
    _apply_endpoint_section(args, payload, protected)
    _apply_training_export_section(args, payload, path.parent, protected)
    _apply_stage_sections(args, payload, path, protected)
    _apply_stage_args(args, payload, path.parent, protected)
    explicit_stage_states = (
        stage_states if "--stages" not in protected and explicit_stages is None else {}
    )
    _finalize_pipeline(
        args,
        explicit_stage_states=explicit_stage_states,
        preserve_stage_order=workflow_stages is not None and "--stages" not in protected,
    )
    _add_training_export_stage_if_configured(args, protected)
    return _entries_from_payload(payload, config_dir=path.parent)


def expand_directory_entries(
    data_entries: list[DataEntry],
    *,
    pipeline: PipelineName,
) -> list[DataEntry]:
    """Expand local directory media inputs into one entry per media file."""
    extensions = VIDEO_EXTENSIONS if pipeline == "video" else IMAGE_EXTENSIONS
    expanded: list[DataEntry] = []
    for entry in data_entries:
        media_path = Path(entry.media_path).expanduser()
        if "://" in entry.media_path or not media_path.is_dir():
            expanded.append(entry)
            continue

        output_root = Path(entry.data_path).expanduser()
        candidates = sorted(
            path
            for path in media_path.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        expanded.extend(
            entry.model_copy(
                update={
                    "id": f"{entry.id or path.name}:{path.name}",
                    "media_path": str(path),
                    "data_path": str(output_root / path.name),
                }
            )
            for path in candidates
        )
    return expanded


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(raw)
    else:
        loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        msg = f"Cookbook config must be a JSON/YAML object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def explicit_cli_options() -> set[str]:
    explicit: set[str] = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            explicit.add(arg.split("=", 1)[0])
    return explicit


def _finalize_pipeline(
    args: Namespace,
    *,
    explicit_stage_states: dict[StageName, bool],
    preserve_stage_order: bool = False,
) -> None:
    pipeline = _pipeline_from_args(args)
    args.pipeline = pipeline
    if args.stages is None:
        args.stages = list(default_stages_for_pipeline(pipeline))

    stages = list(_stage_list(args.stages) or ())
    if preserve_stage_order and not explicit_stage_states:
        args.stages = stages
        return

    selected = set(stages)
    for stage, enabled in explicit_stage_states.items():
        if enabled:
            selected.add(stage)
        else:
            selected.discard(stage)
    args.stages = [stage for stage in SELECTABLE_STAGES if stage in selected]


def _add_training_export_stage_if_configured(args: Namespace, protected: set[str]) -> None:
    if "--stages" in protected or not getattr(args, "training_export_formats", None):
        return
    stages = list(_stage_list(args.stages) or ())
    if "training_export" in stages:
        return
    selected = {*stages, "training_export"}
    args.stages = [stage for stage in SELECTABLE_STAGES if stage in selected]
    workflow_nodes = list(getattr(args, "workflow_nodes", ()) or ())
    if not workflow_nodes:
        return
    node_ids = {node.node_id for node in workflow_nodes}
    node_id = "training_export"
    suffix = 2
    while node_id in node_ids:
        node_id = f"training_export_{suffix}"
        suffix += 1
    workflow_nodes.append(
        WorkflowNode(
            node_id=node_id,
            stage="training_export",
            needs=(workflow_nodes[-1].node_id,),
        )
    )
    args.workflow_nodes = workflow_nodes


def _apply_runtime_section(
    args: Namespace,
    payload: dict[str, Any],
    config_dir: Path,
    protected: set[str],
) -> None:
    legacy_pipeline_settings = _mapping(payload.get("pipeline"))
    runtime = {**legacy_pipeline_settings, **_mapping(payload.get("runtime"))}
    if not runtime:
        return
    _set_arg(
        args,
        "model_cache_path",
        _optional_resolved_path(runtime.get("model_cache_path"), config_dir),
        "--model-cache-path",
        protected,
    )
    _set_arg(args, "gpu_ids", _optional_str(runtime.get("gpu_ids")), "--gpu-ids", protected)


def _apply_container_section(
    args: Namespace,
    payload: dict[str, Any],
    protected: set[str],
) -> None:
    """Apply an optional cookbook ``container:`` block to runner container args.

    Lets a cookbook declare its container wiring once (user, per-stage images,
    environment, bind mounts) so end users only need ``--cookbook-file``.
    Explicit CLI flags always win; ``env`` and ``mounts`` are additive.
    """
    container = _mapping(payload.get("container"))
    if not container:
        return
    _set_arg(
        args,
        "container_user",
        _optional_str(container.get("user")),
        "--container-user",
        protected,
    )
    for stage_key, image in _mapping(container.get("images")).items():
        image_value = _optional_str(image)
        stages = _stage_list([stage_key])
        if image_value is None or not stages:
            continue
        attr, flag = _IMAGE_ARG_BY_STAGE[stages[0]]
        _set_arg(args, attr, image_value, flag, protected)
    # env and mounts are additive append-flags, so cookbook values are merged
    # with (not replaced by) CLI values. Cookbook entries go first so any CLI
    # --container-env for the same name wins (the container runtime keeps the
    # last --env for a given name).
    env_items = _container_env_items(container.get("env"))
    if env_items:
        args.container_env = [*env_items, *args.container_env]
    mount_items = _container_mount_items(container.get("mounts"))
    if mount_items:
        args.container_mount = [*mount_items, *args.container_mount]


def _container_env_items(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}={item}" for key, item in value.items() if str(key)]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _container_mount_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _apply_endpoint_section(
    args: Namespace,
    payload: dict[str, Any],
    protected: set[str],
) -> None:
    endpoints = _mapping(payload.get("endpoints"))
    vlm = _mapping(endpoints.get("vlm")) if endpoints else {}
    llm = _mapping(endpoints.get("llm")) if endpoints else {}
    _set_arg(
        args, "vlm_endpoint_url", _optional_str(vlm.get("url")), "--vlm-endpoint-url", protected
    )
    _set_arg(args, "vlm_model", _optional_str(vlm.get("model")), "--vlm-model", protected)
    _set_arg(
        args, "llm_endpoint_url", _optional_str(llm.get("url")), "--llm-endpoint-url", protected
    )
    _set_arg(args, "llm_model", _optional_str(llm.get("model")), "--llm-model", protected)


def _apply_training_export_section(
    args: Namespace,
    payload: dict[str, Any],
    config_dir: Path,
    protected: set[str],
) -> None:
    export = _mapping(payload.get("training_export"))
    if not export:
        return
    if not _bool(export.get("enabled"), default=True):
        return

    formats = export.get("formats") or export.get("format")
    if "--training-export-format" not in protected and formats is not None:
        raw_formats = formats if isinstance(formats, list) else [formats]
        args.training_export_formats = [str(item) for item in raw_formats if str(item)]

    raw_output_dir = _optional_str(export.get("output_dir") or export.get("out_dir"))
    resolved_output_dir = (
        _resolve_path(raw_output_dir, config_dir=config_dir, must_exist=False)
        if raw_output_dir is not None
        else None
    )
    _set_arg(
        args,
        "training_export_dir",
        _optional_str(resolved_output_dir),
        "--training-export-dir",
        protected,
    )

    task_types = export.get("task_types") or export.get("tasks")
    if "--training-export-task" not in protected and isinstance(task_types, list):
        args.training_export_tasks = [str(item) for item in task_types if str(item)]

    metadata = _mapping(export.get("metadata"))
    description = _optional_str(export.get("description") or metadata.get("description"))
    _set_arg(
        args,
        "training_export_description",
        description,
        "--training-export-description",
        protected,
    )
    license_value = _optional_str(export.get("license") or metadata.get("license"))
    _set_arg(
        args,
        "training_export_license",
        license_value,
        "--training-export-license",
        protected,
    )

    tags = export.get("tags") or metadata.get("tags")
    if "--training-export-tag" not in protected and isinstance(tags, list):
        args.training_export_tags = [str(item) for item in tags if str(item)]

    if "--training-export-no-copy-media" not in protected and "copy_media" in export:
        args.training_export_no_copy_media = not _bool(export.get("copy_media"), default=True)
    if "--training-export-emit-media-root-as-null" not in protected:
        args.training_export_emit_media_root_as_null = _bool(
            export.get("emit_media_root_as_null"),
            default=args.training_export_emit_media_root_as_null,
        )


def _apply_stage_sections(
    args: Namespace,
    payload: dict[str, Any],
    config_path: Path,
    protected: set[str],
) -> None:
    config_dir = config_path.parent
    super_resolution = _mapping(payload.get("super_resolution"))
    detection = _mapping(payload.get("detection_and_tracking"))
    mcq_generation = _mapping(payload.get("mcq_generation"))
    visual_qa = _mapping(payload.get("visual_qa"))

    _set_arg(
        args,
        "sr_resolver",
        _optional_str(super_resolution.get("resolver") or super_resolution.get("model")),
        "--sr-resolver",
        protected,
    )
    _set_arg(
        args,
        "sr_variant",
        _optional_str(super_resolution.get("variant")),
        "--sr-variant",
        protected,
    )

    tracker = _tracker_from_detection(detection)
    _set_arg(args, "tracker", tracker, "--tracker", protected)
    classes = detection.get("classes")
    if "--classes" not in protected and isinstance(classes, list):
        args.classes = [str(item) for item in classes]

    question_bank = _question_bank_path(
        config_dir=config_dir,
        mcq_generation=mcq_generation,
        visual_qa=visual_qa,
    )
    _set_arg(
        args,
        "question_bank_file",
        question_bank,
        "--question-bank-file",
        protected,
    )
    _set_arg(
        args,
        "reasoning_config_file",
        str(config_path),
        "--reasoning-config-file",
        protected,
    )
    _set_arg(
        args,
        "pas_config_file",
        str(config_path),
        "--pas-config-file",
        protected,
    )


def _apply_stage_args(
    args: Namespace,
    payload: dict[str, Any],
    config_dir: Path,
    protected: set[str],
) -> None:
    if "--stage-arg" in protected:
        return
    stage_args = _mapping(payload.get("stage_args"))
    if not stage_args:
        return
    raw_args = list(args.stage_arg)
    for stage_name, values in stage_args.items():
        stage = _stage_list([stage_name])
        if not stage:
            continue
        if isinstance(values, list):
            raw_args.extend(
                f"{stage[0]}={value}"
                for value in _resolve_stage_arg_values(values, config_dir=config_dir)
            )
        elif values is not None:
            raw_args.extend(
                f"{stage[0]}={value}"
                for value in _resolve_stage_arg_values([values], config_dir=config_dir)
            )
    args.stage_arg = raw_args


def _entries_from_payload(payload: dict[str, Any], *, config_dir: Path) -> list[DataEntry]:
    raw_entries = payload.get("data")
    if not isinstance(raw_entries, list):
        return []

    entries: list[DataEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry)
        inputs = _mapping(entry.get("inputs"))
        output = _mapping(entry.get("output"))
        media_path = _optional_str(inputs.get("media_path") or entry.get("media_path"))
        data_path = _optional_str(
            output.get("out_dir")
            or output.get("data_path")
            or entry.get("data_path")
            or entry.get("out_dir")
        )
        if media_path is None or data_path is None:
            keys = sorted(entry) if entry else []
            msg = (
                f"Cookbook data entry {index} must define media_path and data_path/out_dir; "
                f"keys={keys}, raw={raw_entry!r}"
            )
            raise ValueError(msg)
        entry_id = _optional_str(entry.get("id"))
        values = {
            "media_path": _resolve_path(media_path, config_dir=config_dir, must_exist=False),
            "data_path": _resolve_path(data_path, config_dir=config_dir, must_exist=False),
        }
        if entry_id is not None:
            values["id"] = entry_id
        entries.append(DataEntry(**values))
    return entries


def _stage_enabled_map(payload: dict[str, Any]) -> dict[StageName, bool]:
    mapping = {
        "super_resolution": "super_resolution",
        "detection_and_tracking": "detection_and_tracking",
        "referring_expressions": "referring_expressions",
        "grounding_2d": "grounding_2d",
        "captioning": "captioning",
        "vlm_json": "captioning",
        "dense_caption": "captioning",
        "visual_qa": "visual_qa",
        "reasoning": "reasoning",
        # Legacy cookbook section name kept for backward compatibility.
        "daft_export": "reasoning",
        "person_attribute_search": "person_attribute_search",
    }
    configured: dict[StageName, list[tuple[str, bool]]] = {}
    for section_name, stage_name in mapping.items():
        section = _mapping(payload.get(section_name))
        enabled = section.get("enabled")
        if isinstance(enabled, bool):
            configured.setdefault(cast(StageName, stage_name), []).append((section_name, enabled))

    states: dict[StageName, bool] = {}
    for stage_name, values in configured.items():
        enabled_values = {enabled for _, enabled in values}
        if len(enabled_values) > 1:
            rendered = ", ".join(f"{section}={enabled}" for section, enabled in values)
            msg = f"Conflicting enabled values for stage {stage_name}: {rendered}"
            raise ValueError(msg)
        states[stage_name] = values[0][1]
    return states


def _stage_list(value: object) -> tuple[StageName, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        msg = f"stages must be a list, got {type(value).__name__}"
        raise ValueError(msg)
    stages: list[StageName] = []
    for item in value:
        normalized = str(item).strip().replace("-", "_")
        normalized = STAGE_ALIASES.get(normalized, normalized)
        for choice in SELECTABLE_STAGES:
            if normalized == choice:
                stages.append(choice)
                break
        else:
            choices = ", ".join(SELECTABLE_STAGES)
            msg = f"Invalid stage {item!r}; expected one of: {choices}."
            raise ValueError(msg)
    return tuple(stages)


def _workflow_node_list(
    payload: dict[str, Any], *, config_dir: Path
) -> tuple[WorkflowNode, ...] | None:
    workflow = _mapping(payload.get("workflow"))
    if not workflow:
        return None
    nodes_value = workflow.get("nodes")
    if nodes_value is None:
        return None
    nodes = _workflow_nodes(nodes_value, config_dir=config_dir)
    if not nodes:
        msg = "workflow.nodes must contain at least one node"
        raise ValueError(msg)
    ordered_nodes = _topological_workflow_nodes(nodes)
    return ordered_nodes


def _workflow_nodes(value: object, *, config_dir: Path) -> tuple[WorkflowNode, ...]:
    if isinstance(value, dict):
        return tuple(
            _workflow_node_from_mapping(str(node_id), raw, config_dir=config_dir)
            for node_id, raw in value.items()
        )
    if isinstance(value, list):
        nodes: list[WorkflowNode] = []
        for index, raw in enumerate(value):
            node = _mapping(raw)
            node_id = _optional_str(node.get("id"))
            if node_id is None:
                msg = f"workflow.nodes[{index}] requires non-empty id"
                raise ValueError(msg)
            nodes.append(_workflow_node_from_mapping(node_id, raw, config_dir=config_dir))
        return tuple(nodes)
    msg = f"workflow.nodes must be a mapping or list, got {type(value).__name__}"
    raise ValueError(msg)


def _workflow_node_from_mapping(node_id: str, raw: object, *, config_dir: Path) -> WorkflowNode:
    normalized_node_id = node_id.strip()
    if not normalized_node_id:
        msg = "workflow node id must be non-empty"
        raise ValueError(msg)
    if isinstance(raw, str):
        stage_value: object = raw
        needs_value: object = ()
        args_value: object = ()
    else:
        node = _mapping(raw)
        if not node:
            msg = f"workflow node {node_id!r} must be an object or stage string"
            raise ValueError(msg)
        stage_value = node.get("stage")
        needs_value = node.get("needs", node.get("depends_on", ()))
        args_value = node.get("args", ())
    stage = _stage_from_value(stage_value, context=f"workflow node {normalized_node_id!r}")
    needs = _workflow_needs(needs_value, node_id=normalized_node_id)
    node_args = _workflow_node_args(
        args_value,
        node_id=normalized_node_id,
        config_dir=config_dir,
    )
    return WorkflowNode(node_id=normalized_node_id, stage=stage, needs=needs, args=node_args)


def _workflow_node_args(value: object, *, node_id: str, config_dir: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        msg = f"workflow node {node_id!r} args must be a list"
        raise ValueError(msg)
    return _resolve_stage_arg_values(list(value), config_dir=config_dir)


def _stage_from_value(value: object, *, context: str) -> StageName:
    if value is None or str(value).strip() == "":
        msg = f"{context} requires a stage"
        raise ValueError(msg)
    stages = _stage_list([value])
    if not stages:
        msg = f"{context} requires a stage"
        raise ValueError(msg)
    return stages[0]


def _workflow_needs(value: object, *, node_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        msg = f"workflow node {node_id!r} needs must be a node id or list of node ids"
        raise ValueError(msg)
    needs = tuple(str(item).strip() for item in values if str(item).strip())
    if node_id in needs:
        msg = f"workflow node {node_id!r} cannot depend on itself"
        raise ValueError(msg)
    return needs


def _topological_workflow_nodes(nodes: tuple[WorkflowNode, ...]) -> tuple[WorkflowNode, ...]:
    by_id: dict[str, WorkflowNode] = {}
    for node in nodes:
        if node.node_id in by_id:
            msg = f"Duplicate workflow node id {node.node_id!r}"
            raise ValueError(msg)
        by_id[node.node_id] = node

    for node in nodes:
        missing = [dep for dep in node.needs if dep not in by_id]
        if missing:
            rendered = ", ".join(missing)
            msg = f"workflow node {node.node_id!r} depends on unknown node(s): {rendered}"
            raise ValueError(msg)

    ordered: list[WorkflowNode] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: WorkflowNode) -> None:
        if node.node_id in visited:
            return
        if node.node_id in visiting:
            msg = f"workflow.nodes contains a dependency cycle at {node.node_id!r}"
            raise ValueError(msg)
        visiting.add(node.node_id)
        for dep in node.needs:
            visit(by_id[dep])
        visiting.remove(node.node_id)
        visited.add(node.node_id)
        ordered.append(node)

    for node in nodes:
        visit(node)
    return tuple(ordered)


def _optional_pipeline(value: object) -> PipelineName | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("type") or value.get("kind") or value.get("name") or value.get("mode")
        if value is None:
            return None
    rendered = str(value).strip().lower()
    if rendered not in {"video", "image"}:
        msg = f"pipeline must be 'video' or 'image', got {value!r}"
        raise ValueError(msg)
    return cast(PipelineName, rendered)


def _pipeline_from_payload(payload: dict[str, Any], config_path: Path) -> PipelineName | None:
    pipeline = _optional_pipeline(payload.get("pipeline"))
    if pipeline is not None:
        return pipeline
    return _pipeline_from_filename(config_path)


def _pipeline_from_filename(config_path: Path) -> PipelineName | None:
    name = config_path.stem.lower()
    if "image" in name:
        return "image"
    if "video" in name:
        return "video"
    return None


def _pipeline_from_args(args: Namespace) -> PipelineName:
    return _optional_pipeline(getattr(args, "pipeline", None)) or DEFAULT_PIPELINE


def _tracker_from_detection(detection: dict[str, Any]) -> str | None:
    model = _optional_str(detection.get("model"))
    tracker = _optional_str(detection.get("tracker"))
    if model == "rfdetr" and tracker:
        return f"rfdetr-{tracker}"
    if tracker:
        return tracker
    return model


def _question_bank_path(
    *,
    config_dir: Path,
    mcq_generation: dict[str, Any],
    visual_qa: dict[str, Any],
) -> str | None:
    source = _mapping(mcq_generation.get("window_metadata_extraction"))
    value = (
        _optional_str(visual_qa.get("question_bank_file"))
        or _optional_str(source.get("question_bank_file"))
        or _optional_str(mcq_generation.get("question_bank_file"))
    )
    if value is None:
        return None
    return _resolve_path(value, config_dir=config_dir, must_exist=False)


def _optional_resolved_path(value: object, config_dir: Path) -> str | None:
    rendered = _optional_str(value)
    if rendered is None:
        return None
    return _resolve_path(rendered, config_dir=config_dir, must_exist=False)


def _optional_config_relative_path(value: object, config_dir: Path) -> str | None:
    rendered = _optional_str(value)
    if rendered is None:
        return None
    path = Path(rendered).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def _resolve_path(value: str, *, config_dir: Path, must_exist: bool) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        if must_exist and not path.exists():
            msg = f"Path does not exist: {value}"
            raise FileNotFoundError(msg)
        return str(path)
    candidates = tuple((root / path).resolve() for root in _path_roots(config_dir))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if must_exist:
        msg = f"Path does not exist: {value}"
        raise FileNotFoundError(msg)
    if _is_repo_root_style_path(path):
        repo_root = _repo_root_candidate(config_dir)
        if repo_root is not None:
            return str((repo_root / path).resolve())
        return str((Path.cwd() / path).resolve())
    return str(candidates[0])


def _path_roots(config_dir: Path) -> tuple[Path, ...]:
    roots = [config_dir]
    if config_dir.name == "configs":
        roots.append(config_dir.parent)
    repo_root = _repo_root_candidate(config_dir)
    if repo_root is not None:
        roots.append(repo_root)
    roots.extend([Path.cwd(), Path.cwd() / "services" / "workflow_runner"])
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def _repo_root_candidate(config_dir: Path) -> Path | None:
    start = config_dir.resolve()
    candidate: Path | None = None
    for ancestor in (start, *start.parents):
        if (ancestor / ".git").exists():
            return ancestor
        if _has_repo_marker(ancestor):
            candidate = ancestor
    return candidate


def _has_repo_marker(path: Path) -> bool:
    return any(
        (path / marker).exists()
        for marker in (".git", "pyproject.toml", "setup.cfg", "requirements.txt")
    )


def _is_repo_root_style_path(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] not in {".", ".."}


def _resolve_stage_arg_values(values: list[object], *, config_dir: Path) -> tuple[str, ...]:
    resolved: list[str] = []
    expect_path = False
    for value in values:
        rendered = str(value)
        if expect_path:
            resolved.append(_resolve_path(rendered, config_dir=config_dir, must_exist=False))
            expect_path = False
            continue
        inline = _resolve_inline_path_stage_arg(rendered, config_dir=config_dir)
        if inline is not None:
            resolved.append(inline)
            continue
        resolved.append(rendered)
        expect_path = rendered in PATH_STAGE_ARG_FLAGS
    return tuple(resolved)


def _resolve_inline_path_stage_arg(value: str, *, config_dir: Path) -> str | None:
    for flag in PATH_STAGE_ARG_FLAGS:
        prefix = f"{flag}="
        if value.startswith(prefix):
            path = _resolve_path(value[len(prefix) :], config_dir=config_dir, must_exist=False)
            return f"{flag}={path}"
    return None


def _set_arg(
    args: Namespace,
    attr: str,
    value: str | None,
    flag: str,
    protected: set[str],
) -> None:
    if value is None or flag in protected:
        return
    setattr(args, attr, value)


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    return rendered if rendered else None


def _bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    rendered = str(value).strip().lower()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    return default


__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "WorkflowNode",
    "apply_cookbook_config",
    "expand_directory_entries",
    "explicit_cli_options",
]
