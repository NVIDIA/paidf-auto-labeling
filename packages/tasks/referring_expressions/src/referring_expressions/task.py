# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Referring-expressions task: DAFT boxes → VLM region phrases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import (
    DataEntry,
    ensure_scene_skeleton,
    is_image_path,
    read_json,
    read_pipeline_state,
    write_json,
    write_pipeline_state,
)
from core.exceptions import InvalidInputError
from core.tasks import SequentialTask
from PIL import Image

from referring_expressions.artifacts import (
    REFERRING_EXPRESSIONS_ARTIFACTS_KEY,
    ReferringExpressionsArtifactsState,
)
from referring_expressions.clients import (
    ChatRequest,
    ReferringEndpointClient,
    create_endpoint_client,
)
from referring_expressions.config import ReferringExpressionsConfig
from referring_expressions.media import read_image_payload
from referring_expressions.overlay import draw_marked_boxes
from referring_expressions.parsing import normalize_bbox, parse_json_array
from referring_expressions.prompts import greedy_match_by_iou, region_expression_prompt
from referring_expressions.types import normalize_object_type


class ReferringExpressionsTask(SequentialTask):
    """Generate discriminative region phrases for known instance boxes."""

    def __init__(
        self,
        *,
        config: ReferringExpressionsConfig | None = None,
        client: ReferringEndpointClient | None = None,
        name: str = "referring_expressions",
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)
        self.config = config or ReferringExpressionsConfig()
        self.client = client or create_endpoint_client(
            provider=self.config.vlm_provider,
            endpoint_url=self.config.vlm_endpoint_url,
            model=self.config.vlm_model,
            timeout_s=self.config.timeout_s,
            retries=self.config.retries,
            retry_backoff_s=self.config.retry_backoff_s,
        )

    def run(self, data_entry: DataEntry) -> DataEntry:
        """Run Step 0 region expressions for one image DataEntry."""
        if not self.config.enabled:
            return data_entry
        image_path = Path(data_entry.media_path)
        if not is_image_path(image_path):
            raise InvalidInputError(
                f"Referring expressions MVP requires an image input: {image_path}"
            )
        if not image_path.exists():
            raise InvalidInputError(f"Referring expressions image missing: {image_path}")

        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        sidecar_dir = scene_paths.sidecars_dir / "referring_expressions"
        # Stage deliverable (not a registered DAFT task type): keep under sidecars/
        # so detection-stage DAFT validation can rerun without rejecting this file.
        final_path = sidecar_dir / "referring_expressions.json"
        region_path = sidecar_dir / "step0_region_expressions.json"
        objects_path = scene_paths.contextual_dir / "objects.json"
        if not objects_path.exists():
            raise InvalidInputError(
                "Referring expressions requires contextual/objects.json from "
                "detection_and_tracking or grounding_2d."
            )
        width, height = _image_size(image_path)
        box_records, frame_found = _boxes_for_frame(
            read_json(objects_path),
            frame_number=self.config.frame_number,
            width=width,
            height=height,
        )
        if not frame_found:
            raise InvalidInputError(
                f"Referring expressions found no frame_number="
                f"{self.config.frame_number} in contextual/objects.json."
            )
        if final_path.exists() and not self.config.force_reprocess:
            final_payload = read_json(final_path)
            region_count = _region_count(final_payload)
            if region_count == len(box_records):
                self._record_state(
                    data_entry=data_entry,
                    success=True,
                    final_path=final_path,
                    region_path=region_path if region_path.exists() else None,
                    final_payload=final_payload,
                )
                return data_entry
            self._record_state(
                data_entry=data_entry,
                success=False,
                final_path=final_path,
                region_path=region_path if region_path.exists() else None,
                final_payload=final_payload,
            )
            self.logger.warning(
                "Ignoring stale referring-expressions cache at %s: %s",
                final_path,
                _incomplete_regions_message(region_count, len(box_records)),
            )

        if not box_records:
            payload = _empty_payload(image_path=image_path, width=width, height=height)
            write_json(region_path, payload)
            write_json(final_path, payload)
            self._record_state(
                data_entry=data_entry,
                success=True,
                final_path=final_path,
                region_path=region_path,
                final_payload=payload,
            )
            return data_entry

        vlm_image = image_path
        overlay_path = sidecar_dir / "marked_boxes.jpg"
        if self.config.draw_box_overlay:
            vlm_image = draw_marked_boxes(image_path, box_records, overlay_path)

        response = self.client.generate(
            ChatRequest(
                prompt=region_expression_prompt(box_records),
                media=(read_image_payload(vlm_image),),
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )
        regions = _regions_from_response(
            response,
            box_records=box_records,
            width=width,
            height=height,
            min_iou=self.config.min_match_iou,
        )
        payload = {
            "schema_version": "1",
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "overlay_path": str(overlay_path) if self.config.draw_box_overlay else None,
            "regions": regions,
        }
        write_json(region_path, payload)
        write_json(final_path, payload)
        if len(regions) != len(box_records):
            self._record_state(
                data_entry=data_entry,
                success=False,
                final_path=final_path,
                region_path=region_path,
                final_payload=payload,
            )
            raise InvalidInputError(_incomplete_regions_message(len(regions), len(box_records)))
        self._record_state(
            data_entry=data_entry,
            success=True,
            final_path=final_path,
            region_path=region_path,
            final_payload=payload,
        )
        return data_entry

    def _record_state(
        self,
        *,
        data_entry: DataEntry,
        success: bool,
        final_path: Path,
        region_path: Path | None,
        final_payload: dict[str, Any],
    ) -> None:
        pipeline_state = read_pipeline_state(data_entry.data_path)
        pipeline_state.data_entry_id = pipeline_state.data_entry_id or data_entry.id
        pipeline_state.media_path = pipeline_state.media_path or data_entry.media_path
        regions = final_payload.get("regions", [])
        pipeline_state.task_artifacts[REFERRING_EXPRESSIONS_ARTIFACTS_KEY] = (
            ReferringExpressionsArtifactsState(
                success=success,
                artifact_json=str(final_path),
                region_json=str(region_path) if region_path else None,
                region_count=len(regions) if isinstance(regions, list) else 0,
            ).model_dump()
        )
        write_pipeline_state(data_entry.data_path, pipeline_state)


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _empty_payload(*, image_path: Path, width: int, height: int) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "overlay_path": None,
        "regions": [],
    }


def _region_count(payload: dict[str, Any]) -> int:
    regions = payload.get("regions", [])
    return len(regions) if isinstance(regions, list) else 0


def _incomplete_regions_message(region_count: int, box_count: int) -> str:
    return (
        f"Referring expressions matched {region_count}/{box_count} authoritative boxes; "
        "refusing to mark the run successful."
    )


def _boxes_for_frame(
    objects_payload: dict[str, Any],
    *,
    frame_number: int,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    frame_found = False
    for frame in objects_payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        if int(frame.get("frame_number", 0)) != frame_number:
            continue
        frame_found = True
        for instance in frame.get("instances", []):
            if not isinstance(instance, dict):
                continue
            bbox = normalize_bbox(
                instance.get("bounding_box_2d_tight"),
                width=width,
                height=height,
            )
            if bbox is None:
                continue
            object_id = instance.get("object_id")
            records.append(
                {
                    "object_id": str(object_id) if object_id is not None else None,
                    "bbox": bbox,
                    "mark": len(records) + 1,
                }
            )
    return records, frame_found


def _regions_from_response(
    response: str,
    *,
    box_records: list[dict[str, Any]],
    width: int,
    height: int,
    min_iou: float,
) -> list[dict[str, Any]]:
    parsed = parse_json_array(response)
    predicted: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        bbox = normalize_bbox(item.get("bbox_2d"), width=width, height=height)
        mark_raw = item.get("mark")
        try:
            mark = int(mark_raw) if mark_raw is not None else None
        except (TypeError, ValueError):
            mark = None
        predicted.append(
            {
                "mark": mark,
                "bbox": bbox,
                "type": normalize_object_type(item.get("type")),
                "color": str(item.get("color") or "unknown").strip().lower() or "unknown",
                "description": description,
            }
        )

    by_mark = {int(r["mark"]): r for r in box_records if r.get("mark") is not None}
    used_ids: set[str] = set()
    regions: list[dict[str, Any]] = []
    unmatched_preds: list[dict[str, Any]] = []

    for pred in predicted:
        matched: dict[str, Any] | None = None
        mark = pred.get("mark")
        if isinstance(mark, int) and mark in by_mark:
            candidate = by_mark[mark]
            oid = candidate.get("object_id")
            if oid is None or str(oid) not in used_ids:
                matched = candidate
        if matched is not None:
            oid = matched.get("object_id")
            if oid is not None:
                used_ids.add(str(oid))
            regions.append(_region_payload(pred, matched))
        else:
            unmatched_preds.append(pred)

    remaining = [
        record
        for record in box_records
        if record.get("object_id") is None or str(record.get("object_id")) not in used_ids
    ]
    iou_preds = [pred for pred in unmatched_preds if pred.get("bbox") is not None]
    for pred, matched in greedy_match_by_iou(
        predicted=iou_preds,
        candidates=remaining,
        min_iou=min_iou,
    ):
        if matched is None:
            continue
        oid = matched.get("object_id")
        if oid is not None:
            used_ids.add(str(oid))
        regions.append(_region_payload(pred, matched))

    mark_order = {r.get("object_id"): i for i, r in enumerate(box_records)}
    regions.sort(key=lambda region: mark_order.get(region.get("object_id"), 10_000))
    return regions


def _region_payload(pred: dict[str, Any], matched: dict[str, Any] | None) -> dict[str, Any]:
    if matched and matched.get("bbox"):
        out_bbox = list(matched["bbox"])
        object_id = matched.get("object_id")
        mark = matched.get("mark")
    else:
        out_bbox = list(pred["bbox"]) if pred.get("bbox") else [0, 0, 0, 0]
        object_id = None
        mark = pred.get("mark")
    return {
        "object_id": object_id,
        "mark": mark,
        "bbox": out_bbox,
        "type": pred["type"],
        "color": pred["color"],
        "description": pred["description"],
    }


__all__ = ["ReferringExpressionsTask"]
