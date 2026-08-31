# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
import pytest
from detection_and_tracking.backends import rfdetr
from detection_and_tracking.backends.rfdetr import _BoostTrackAdapter, _RFDetrRuntime
from detection_and_tracking.config import DetectionAndTrackingConfig


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.offset = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resolve_rfdetr_checkpoint_accepts_existing_verified_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf-detr-base.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("RFDETR_MODEL_PATH", str(checkpoint))
    monkeypatch.setattr(rfdetr, "RFDETR_BASE_SHA256", _sha256(b"checkpoint"))

    resolved = rfdetr._resolve_rfdetr_checkpoint(
        DetectionAndTrackingConfig(allow_model_download=False),
        logger=logging.getLogger("test_rfdetr"),
    )

    assert resolved == checkpoint.resolve()


def test_resolve_rfdetr_checkpoint_removes_failed_checksum_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "models"
    checkpoint = cache_root / "rfdetr" / "rf-detr-base.pth"
    monkeypatch.delenv("RFDETR_MODEL_PATH", raising=False)
    monkeypatch.setattr(rfdetr, "RFDETR_BASE_SHA256", _sha256(b"expected"))
    monkeypatch.setattr(rfdetr, "resolve_model_cache_root", lambda _: cache_root)
    monkeypatch.setattr(
        rfdetr.urllib.request,
        "urlopen",
        lambda url, timeout: _Response(b"partial"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rfdetr._resolve_rfdetr_checkpoint(
            DetectionAndTrackingConfig(allow_model_download=True),
            logger=logging.getLogger("test_rfdetr"),
        )

    assert not checkpoint.exists()
    assert list(checkpoint.parent.glob(f".{checkpoint.name}.*.tmp")) == []


def test_boosttrack_adapter_advances_tracker_on_empty_detections() -> None:
    adapter = object.__new__(_BoostTrackAdapter)
    adapter.np = _FakeNumpy()
    tracker = _CaptureBoostTrack()
    adapter._tracker = tracker
    frame = object()

    result = adapter.update(_FakeArray((0, 6)), frame)

    assert result.shape == (0, 7)
    assert tracker.seen_detections is not None
    assert tracker.seen_detections.shape == (0, 5)
    assert tracker.seen_frame is frame


def test_boosttrack_adapter_honors_per_class_trackers() -> None:
    adapter = object.__new__(_BoostTrackAdapter)
    adapter.np = np
    adapter.per_class = True
    adapter._tracker_cls = _CaptureClassBoostTrack
    adapter._tracker_kwargs = {}
    adapter._trackers_by_class = {}
    _CaptureClassBoostTrack.created = []
    frame = object()
    detections = np.array(
        [
            [0.0, 0.0, 10.0, 10.0, 0.9, 2.0],
            [20.0, 20.0, 30.0, 30.0, 0.8, 0.0],
        ],
        dtype=float,
    )

    tracks = adapter.update(detections, frame)

    assert set(adapter._trackers_by_class) == {0, 2}
    assert tracks.shape == (2, 7)
    assert set(tracks[:, 6].astype(int)) == {0, 2}

    adapter.update(np.array([[21.0, 21.0, 31.0, 31.0, 0.7, 0.0]], dtype=float), frame)

    missing_class_tracker = adapter._trackers_by_class[2]
    assert missing_class_tracker.calls[-1].shape == (0, 5)


def test_boosttrack_adapter_uses_configurable_lifecycle_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Module:
        BoostTrack = _CaptureConfiguredBoostTrack

    def fake_import_optional(name: str, *, backend: str) -> object:
        _ = backend
        return np if name == "numpy" else _Module

    monkeypatch.setattr(rfdetr, "import_optional", fake_import_optional)

    _BoostTrackAdapter(
        config=DetectionAndTrackingConfig(
            threshold=0.2,
            iou_threshold=0.4,
            min_hits=5,
            max_age=90,
        ),
    )

    assert _CaptureConfiguredBoostTrack.kwargs == {
        "det_thresh": 0.2,
        "iou_threshold": 0.4,
        "max_age": 90,
        "min_hits": 5,
    }


def test_rfdetr_loose_bbox_matches_legacy_expansion() -> None:
    assert _RFDetrRuntime._loose_bbox(
        [512.29, 305.79, 559.14, 444.32],
        width=1280,
        height=720,
    ) == pytest.approx([507.61, 291.94, 563.83, 458.17], abs=0.01)


def test_rfdetr_loose_bbox_honors_custom_expansion_ratio() -> None:
    assert _RFDetrRuntime._loose_bbox(
        [100.0, 100.0, 200.0, 200.0],
        width=1280,
        height=720,
        expansion_ratio=0.0,
    ) == pytest.approx([100.0, 100.0, 200.0, 200.0], abs=0.01)
    assert _RFDetrRuntime._loose_bbox(
        [100.0, 100.0, 200.0, 200.0],
        width=1280,
        height=720,
        expansion_ratio=0.2,
    ) == pytest.approx([80.0, 80.0, 220.0, 220.0], abs=0.01)


def test_rfdetr_loose_bbox_clips_to_image_bounds() -> None:
    assert _RFDetrRuntime._loose_bbox(
        [0.0, 0.0, 10.0, 10.0],
        width=100,
        height=100,
        expansion_ratio=0.5,
    ) == pytest.approx([0.0, 0.0, 15.0, 15.0], abs=0.01)


def test_rfdetr_config_defaults_match_legacy() -> None:
    config = DetectionAndTrackingConfig()
    assert config.threshold == 0.2
    assert config.per_class is True
    assert config.min_hits == 3
    assert config.max_age == 60
    assert config.min_track_frames == 5
    assert config.bbox_expansion_ratio == 0.1


def test_rfdetr_normalizes_dict_coco_classes_by_class_id() -> None:
    runtime = object.__new__(_RFDetrRuntime)
    runtime._coco_classes = rfdetr._normalize_coco_classes({1: "person", 3: "car"})

    assert runtime._class_name(0) == "class_0"
    assert runtime._class_name(1) == "person"
    assert runtime._class_name(2) == "class_2"
    assert runtime._class_name(3) == "car"


def test_rfdetr_normalizes_sequence_coco_classes_as_names() -> None:
    assert rfdetr._normalize_coco_classes(["person", "car"]) == ["person", "car"]


def test_rfdetr_class_filter_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown RF-DETR class name"):
        rfdetr._resolve_class_filter_ids(["person", "car"], ("forklfit",))


class _FakeArray:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.size = shape[0] * shape[1]


class _FakeNumpy:
    def empty(self, shape: tuple[int, int], *, dtype: type[float]) -> _FakeArray:
        _ = dtype
        return _FakeArray(shape)


class _CaptureBoostTrack:
    def __init__(self) -> None:
        self.seen_detections: _FakeArray | None = None
        self.seen_frame: object | None = None

    def update(self, detections: _FakeArray, *, frame_bgr: object) -> _FakeArray:
        self.seen_detections = detections
        self.seen_frame = frame_bgr
        return detections


class _CaptureClassBoostTrack:
    created: list["_CaptureClassBoostTrack"] = []

    def __init__(self) -> None:
        self.track_id = len(self.created) + 1
        self.calls: list[np.ndarray] = []
        self.created.append(self)

    def update(self, detections: np.ndarray, *, frame_bgr: object) -> np.ndarray:
        _ = frame_bgr
        self.calls.append(detections.copy())
        if detections.size == 0:
            return np.empty((0, 6), dtype=float)
        return np.array(
            [
                [
                    detections[0, 0],
                    detections[0, 1],
                    detections[0, 2],
                    detections[0, 3],
                    float(self.track_id),
                    detections[0, 4],
                ]
            ],
            dtype=float,
        )


class _CaptureConfiguredBoostTrack:
    kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).kwargs = kwargs
