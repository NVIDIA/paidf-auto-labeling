# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D grounding task: VLM expressions + SAM3 boxes/masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import (
    DataEntry,
    ScenePaths,
    ensure_scene_skeleton,
    is_image_path,
    read_json,
    read_pipeline_state,
    write_json,
    write_pipeline_state,
)
from core.exceptions import InvalidInputError
from core.tasks import SequentialTask
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.task import DetectionAndTrackingTask
from PIL import Image

from grounding_2d.artifacts import GROUNDING_2D_ARTIFACTS_KEY, Grounding2DArtifactsState
from grounding_2d.clients import ChatRequest, GroundingEndpointClient, create_endpoint_client
from grounding_2d.config import Grounding2DConfig
from grounding_2d.expressions import (
    ExpressionFilterPolicy,
    filter_groundable_expressions,
    fits_sam3_text_encoder,
)
from grounding_2d.media import read_image_payload
from grounding_2d.parsing import normalize_bbox, parse_json_object
from grounding_2d.prompts import expression_prompt


class Grounding2DTask(SequentialTask):
    """Extract referring expressions with a VLM, then ground them with SAM3."""

    def __init__(
        self,
        *,
        config: Grounding2DConfig | None = None,
        client: GroundingEndpointClient | None = None,
        name: str = "grounding_2d",
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)
        self.config = config or Grounding2DConfig()
        self.client = client or create_endpoint_client(
            provider=self.config.vlm_provider,
            endpoint_url=self.config.vlm_endpoint_url,
            model=self.config.vlm_model,
            timeout_s=self.config.timeout_s,
            retries=self.config.retries,
            retry_backoff_s=self.config.retry_backoff_s,
        )

    def run(self, data_entry: DataEntry) -> DataEntry:
        """Run VLM expression extraction then SAM3 text-prompted grounding."""
        if not self.config.enabled:
            return data_entry
        image_path = Path(data_entry.media_path)
        if not is_image_path(image_path):
            raise InvalidInputError(f"2D grounding requires an image input: {image_path}")
        if not image_path.exists():
            raise InvalidInputError(f"2D grounding image input is missing: {image_path}")

        caption = _resolve_caption(data_entry, self.config)
        if not caption:
            raise InvalidInputError(
                "2D grounding requires a caption. Set Grounding2DConfig.caption or write "
                "sidecars/input.json with a caption field."
            )

        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        sidecar_dir = scene_paths.sidecars_dir / "grounding_2d"
        # Stage deliverable (not a registered DAFT task type): keep under sidecars/
        # so detection-stage DAFT validation can rerun without rejecting this file.
        final_path = sidecar_dir / "grounding_2d.json"
        expression_path = sidecar_dir / "step0_expressions.json"
        grounding_path = sidecar_dir / "step1_grounding.json"
        if final_path.exists() and not self.config.force_reprocess:
            self._record_state(
                data_entry=data_entry,
                success=True,
                final_path=final_path,
                expression_path=expression_path if expression_path.exists() else None,
                grounding_path=grounding_path if grounding_path.exists() else None,
                final_payload=read_json(final_path),
            )
            return data_entry

        width, height = _image_size(image_path)
        expression_payload = self._extract_expressions(
            image_path=image_path,
            caption=caption,
        )
        write_json(expression_path, expression_payload)

        final_payload = self._run_sam3_grounding(
            data_entry=data_entry,
            scene_paths=scene_paths,
            image_path=image_path,
            caption=caption,
            width=width,
            height=height,
            expression_payload=expression_payload,
        )
        write_json(grounding_path, final_payload)
        write_json(final_path, final_payload)
        self._record_state(
            data_entry=data_entry,
            success=True,
            final_path=final_path,
            expression_path=expression_path,
            grounding_path=grounding_path,
            final_payload=final_payload,
        )
        return data_entry

    def _extract_expressions(self, *, image_path: Path, caption: str) -> dict[str, Any]:
        response = self.client.generate(
            ChatRequest(
                prompt=expression_prompt(caption),
                media=(read_image_payload(image_path),),
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )
        parsed = parse_json_object(response)
        cleaned_caption = str(parsed.get("cleaned_caption") or caption)
        expressions: list[dict[str, Any]] = []
        for raw_expr in parsed.get("expressions", []):
            if not isinstance(raw_expr, dict):
                continue
            text = str(raw_expr.get("text") or "").strip()
            if not text:
                continue
            expression: dict[str, Any] = {
                "expression_id": f"expr_{len(expressions):05d}",
                "text": text,
                "noun_chunk": raw_expr.get("noun_chunk") or text.split()[-1],
                "groundable": raw_expr.get("groundable", True),
            }
            char_span = _resolve_char_span(
                raw_expr.get("char_span"),
                text=text,
                cleaned_caption=cleaned_caption,
            )
            if char_span is not None:
                expression["char_span"] = char_span
            expressions.append(expression)
        return {
            "schema_version": "1",
            "caption": caption,
            "cleaned_caption": cleaned_caption,
            "expressions": expressions,
        }

    def _run_sam3_grounding(
        self,
        *,
        data_entry: DataEntry,
        scene_paths: ScenePaths,
        image_path: Path,
        caption: str,
        width: int,
        height: int,
        expression_payload: dict[str, Any],
    ) -> dict[str, Any]:
        expressions = list(expression_payload["expressions"])
        if self.config.filter_ungroundable_expressions:
            policy = ExpressionFilterPolicy.empty()
            if self.config.expression_filter_policy_path:
                policy = ExpressionFilterPolicy.from_json_file(
                    self.config.expression_filter_policy_path
                )
            groundable, _skipped = filter_groundable_expressions(expressions, policy=policy)
        else:
            groundable = [
                expr for expr in expressions if fits_sam3_text_encoder(str(expr.get("text") or ""))
            ]
        payload = _base_payload(
            image_path=image_path,
            caption=caption,
            width=width,
            height=height,
            expression_payload=expression_payload,
        )
        prompts = tuple(str(expr["text"]) for expr in groundable)
        if not prompts:
            return payload

        detection_config = DetectionAndTrackingConfig(
            tracker="sam3",
            classes=prompts,
            sam3_prompts=prompts,
            model_cache_path=self.config.sam3_model_cache_path,
            gpu_ids=self.config.sam3_gpu_ids,
            sam3_version=self.config.sam3_version,
            sam3_runtime=self.config.sam3_runtime,
            sam3_target_fps=self.config.sam3_target_fps,
            sam3_session_reset_s=self.config.sam3_session_reset_s,
            sam3_max_duration_s=self.config.sam3_max_duration_s,
            sam3_write_annotated_video=self.config.sam3_write_annotated_media,
            sam3_annotated_video_label_style=self.config.sam3_annotated_media_label_style,
            sam3_annotated_video_mask_opacity=self.config.sam3_annotated_media_mask_opacity,
        )
        DetectionAndTrackingTask(
            config=detection_config,
            name="grounding_2d_sam3",
        ).run(data_entry)

        objects_path = scene_paths.contextual_dir / "objects.json"
        if not objects_path.exists():
            return payload
        objects_payload = read_json(objects_path)
        prompt_to_instances = _sam3_instances_by_prompt(objects_payload)
        groundable_texts = {str(expr["text"]) for expr in groundable}
        bbox_count = 0
        for expr in payload["expressions"]:
            if expr["text"] not in groundable_texts:
                continue
            for instance in prompt_to_instances.get(expr["text"], []):
                bbox = normalize_bbox(
                    instance.get("bounding_box_2d_tight"),
                    width=width,
                    height=height,
                    coordinate_mode="pixel",
                )
                if bbox is None:
                    continue
                area = _bbox_area(bbox)
                if area < self.config.min_bbox_area:
                    continue
                score = _instance_score(instance)
                if score is not None and score < self.config.min_instance_score:
                    continue
                if bbox_count >= self.config.max_instances_per_expression * len(prompts):
                    break
                expr_instances = expr["instances"]
                if len(expr_instances) >= self.config.max_instances_per_expression:
                    break
                contours = instance.get("contours_xy", [])
                expr_instances.append(
                    {
                        "bbox_id": f"sam3_box_{bbox_count:05d}",
                        "bbox": bbox,
                        "bbox_score": score,
                        "mask_id": str(instance.get("object_id") or f"sam3_mask_{bbox_count:05d}"),
                        "segmentation": {
                            "format": "contours_xy",
                            "contours": contours,
                            "size": [height, width],
                        },
                        "area": area,
                        "mask_score": score,
                        "source": "sam3",
                    }
                )
                bbox_count += 1
        return payload

    def _record_state(
        self,
        *,
        data_entry: DataEntry,
        success: bool,
        final_path: Path,
        expression_path: Path | None,
        grounding_path: Path | None,
        final_payload: dict[str, Any],
    ) -> None:
        pipeline_state = read_pipeline_state(data_entry.data_path)
        pipeline_state.data_entry_id = pipeline_state.data_entry_id or data_entry.id
        pipeline_state.media_path = pipeline_state.media_path or data_entry.media_path
        expressions = final_payload.get("expressions", [])
        instance_count = sum(len(expr.get("instances", [])) for expr in expressions)
        has_masks = any(
            bool(instance.get("segmentation"))
            for expr in expressions
            for instance in expr.get("instances", [])
        )
        pipeline_state.task_artifacts[GROUNDING_2D_ARTIFACTS_KEY] = Grounding2DArtifactsState(
            success=success,
            artifact_json=str(final_path),
            final_json=str(final_path),
            expression_json=str(expression_path) if expression_path else None,
            grounding_json=str(grounding_path) if grounding_path else None,
            segmentation_json=str(grounding_path) if grounding_path else None,
            expression_count=len(expressions),
            instance_count=instance_count,
            has_masks=has_masks,
        ).model_dump()
        write_pipeline_state(data_entry.data_path, pipeline_state)


def _resolve_char_span(
    raw_span: object,
    *,
    text: str,
    cleaned_caption: str,
) -> list[int] | None:
    """Return a validated model-provided or caption-derived span; never invent offsets."""
    if isinstance(raw_span, list) and len(raw_span) == 2:
        try:
            start = int(raw_span[0])
            end = int(raw_span[1])
        except (TypeError, ValueError):
            pass
        else:
            if 0 <= start < end <= len(cleaned_caption) and cleaned_caption[start:end] == text:
                return [start, end]
    index = cleaned_caption.find(text)
    if index >= 0 and cleaned_caption.count(text) == 1:
        return [index, index + len(text)]
    return None


def _resolve_caption(data_entry: DataEntry, config: Grounding2DConfig) -> str:
    """Resolve the scene caption for expression extraction.

    Precedence:
    1. ``Grounding2DConfig.caption`` / ``--caption``
    2. Explicit scene input ``sidecars/input.json`` (manual / GT override)
    3. Captioning stage artifacts (``sidecars/captioning/…``, DAFT pivots)
    """
    if config.caption:
        return str(config.caption)

    data_path = Path(data_entry.data_path)
    explicit = _caption_from_json_file(data_path / "sidecars" / config.input_metadata_filename)
    if explicit:
        return explicit

    for relative in (
        "sidecars/captioning/image_captions.json",
        "sidecars/captioning/image_caption.json",
        "contextual/image_captions.json",
        "task/image_captions.json",
    ):
        caption = _caption_from_json_file(data_path / relative)
        if caption:
            return caption
    return ""


def _caption_from_json_file(path: Path) -> str:
    """Extract a caption string from a JSON sidecar if present."""
    if not path.exists():
        return ""
    payload = read_json(path)
    if not isinstance(payload, dict):
        return ""
    for key in ("caption", "text", "description"):
        value = payload.get(key)
        if value:
            return str(value)
    model_output = payload.get("model_output")
    if isinstance(model_output, dict):
        for key in ("caption", "description", "text"):
            value = model_output.get(key)
            if value:
                return str(value)
    return ""


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
        return int(width), int(height)


def _base_payload(
    *,
    image_path: Path,
    caption: str,
    width: int,
    height: int,
    expression_payload: dict[str, Any],
) -> dict[str, Any]:
    expressions = []
    for expr in expression_payload.get("expressions", []):
        expr_out = dict(expr)
        expr_out["instances"] = []
        expressions.append(expr_out)
    return {
        "schema_version": "1",
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "caption": caption,
        "cleaned_caption": expression_payload.get("cleaned_caption", caption),
        "expressions": expressions,
    }


def _sam3_instances_by_prompt(objects_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prompt_to_instances: dict[str, list[dict[str, Any]]] = {}
    for frame in objects_payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        for instance in frame.get("instances", []):
            if not isinstance(instance, dict):
                continue
            prompt = str(instance.get("prompt") or "")
            if not prompt:
                continue
            prompt_to_instances.setdefault(prompt, []).append(instance)
    return prompt_to_instances


def _instance_score(instance: dict[str, Any]) -> float | None:
    raw = instance.get("score", instance.get("detection_score"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _bbox_area(bbox: list[int]) -> int:
    return max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


__all__ = ["Grounding2DTask"]
