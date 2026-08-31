# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import captioning.captioner as captioner_module
import pytest
from captioning.captioner import DenseCaptioner, _safe_sidecar_filename
from captioning.clients import ChatRequest, MediaPayload
from captioning.config import CaptioningConfig
from captioning.media import CaptionWindow, VideoInfo
from captioning.parsing import extract_json_object, extract_window_description
from core import SceneContext, ensure_scene_skeleton


class _StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[ChatRequest] = []
        self.last_call_metadata = {
            "provider": "openai-compatible",
            "model": "test-model",
            "endpoint_url": "http://localhost:8000/v1",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens": 11,
                "total_tokens": 18,
            },
        }

    def generate(self, request: ChatRequest) -> str:
        self.requests.append(request)
        return self.response


class _VideoRejectThenFrameClient:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.last_call_metadata: dict[str, object] = {}

    def generate(self, request: ChatRequest) -> str:
        self.requests.append(request)
        if request.media and request.media[0].mime_type == "video/mp4":
            raise RuntimeError("endpoint rejected video payload")
        self.last_call_metadata = {
            "provider": "openai-compatible",
            "model": "test-model",
            "endpoint_url": "http://localhost:8000/v1",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 3,
                "prompt_tokens": 5,
                "total_tokens": 8,
            },
        }
        return "Frame fallback caption."


def test_extract_json_object_from_fenced_response() -> None:
    raw = 'prefix\n```json\n{"caption": "A loading dock is visible."}\n```\n'

    parsed = extract_json_object(raw)

    assert parsed == {"caption": "A loading dock is visible."}
    assert extract_window_description(raw, parsed) == "A loading dock is visible."


def test_extract_json_object_accepts_trailing_text() -> None:
    parsed = extract_json_object('{"caption": "A loading dock is visible."}\nextra text')

    assert parsed == {"caption": "A loading dock is visible."}


def test_extract_json_object_accepts_prose_prefixed_json() -> None:
    parsed = extract_json_object('Sure, here is JSON: {"caption": "A dock is visible."}')

    assert parsed == {"caption": "A dock is visible."}


def test_dense_captioner_requires_llm_client_when_summary_enabled() -> None:
    config = CaptioningConfig(
        enable_llm_summary=True,
        llm_provider="openai-compatible",
        llm_model="summary-model",
    )

    with pytest.raises(ValueError, match="enable_llm_summary requires an llm_client"):
        DenseCaptioner(config=config, vlm_client=_StaticClient("{}"))


def test_captioning_config_rejects_ambiguous_prompt_sources() -> None:
    with pytest.raises(ValueError, match="prompt_text and prompt_file cannot both be set"):
        CaptioningConfig(prompt_text="inline", prompt_file="prompt.txt")


def test_captioning_config_requires_llm_fields_for_summary() -> None:
    with pytest.raises(ValueError, match="enable_llm_summary requires llm_model"):
        CaptioningConfig(enable_llm_summary=True)


def test_summary_batches_chunk_descriptions_without_dropping_text() -> None:
    llm_client = _StaticClient("short summary")
    captioner = DenseCaptioner(
        config=CaptioningConfig(
            enable_llm_summary=True,
            llm_provider="openai-compatible",
            llm_model="summary-model",
            summary_prompt_text="Summarize.",
            summary_input_token_budget=28,
            max_tokens=8,
            temperature=0.4,
            top_p=0.8,
        ),
        vlm_client=_StaticClient("{}"),
        llm_client=llm_client,
    )

    summary, _metadata = captioner._summarize_windows(
        [
            {
                "start_s": float(idx * 10),
                "end_s": float(idx * 10 + 10),
                "description": f"window-desc-{idx}",
                "inner_chunks": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "description": f"chunk-desc-{idx}",
                    }
                ],
            }
            for idx in range(5)
        ]
    )

    assert summary is not None
    assert "short summary" in summary
    source_prompts = [
        request.prompt for request in llm_client.requests if "chunk-desc-" in request.prompt
    ]
    assert len(source_prompts) > 1
    combined_source_prompts = "\n".join(source_prompts)
    for idx in range(5):
        assert f"chunk-desc-{idx}" in combined_source_prompts
        assert f"window-desc-{idx}" not in combined_source_prompts
    assert "[21.000-22.000s chunk 0] chunk-desc-2" in combined_source_prompts
    for request in llm_client.requests:
        assert request.max_tokens == 8
        assert request.temperature == 0.4
        assert request.top_p == 0.8


def test_dense_captioner_writes_image_outputs(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-jpeg-but-sufficient-for-client-tests")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _StaticClient('```json\n{"caption": "A truck is parked near a warehouse."}\n```')
    captioner = DenseCaptioner(config=CaptioningConfig(), vlm_client=client)

    result = captioner.run(
        media_path=image,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(image),
    )

    assert result.success is True
    assert result.sidecar_json == scene_paths.sidecars_dir / "captioning" / "image_caption.json"
    assert result.image_json == scene_paths.sidecars_dir / "captioning" / "image_captions.json"
    assert len(client.requests) == 1
    assert client.requests[0].media[0].mime_type == "image/jpeg"

    sidecar = json.loads(result.sidecar_json.read_text())
    image_payload = json.loads(result.image_json.read_text())
    assert sidecar["schema_version"] == "1"
    assert sidecar["artifact_type"] == "image_caption_sidecar"
    assert sidecar["media"]["kind"] == "image"
    assert sidecar["media"]["filename"] == "frame.jpg"
    assert sidecar["model"]["provider"] == "openai-compatible"
    assert sidecar["generation"]["max_tokens"] == 1024
    assert sidecar["prompt"]["source"] == "default"
    assert sidecar["caption_status"] == "success"
    assert sidecar["caption"] == "A truck is parked near a warehouse."
    assert image_payload["caption"] == "A truck is parked near a warehouse."


def test_dense_captioner_sends_image_group_in_one_request(tmp_path: Path) -> None:
    group = tmp_path / "views"
    (group / "nested").mkdir(parents=True)
    (group / "z.JPG").write_bytes(b"third")
    (group / "a.png").write_bytes(b"first")
    (group / "nested" / "b.webp").write_bytes(b"second")
    (group / "ignore.txt").write_text("not media", encoding="utf-8")
    representative = group / "a.png"
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _StaticClient('{"caption": "One person shown from several viewpoints."}')
    captioner = DenseCaptioner(
        config=CaptioningConfig(image_group_dir=str(group)),
        vlm_client=client,
    )

    result = captioner.run(
        media_path=representative,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(representative),
    )

    assert result.success is True
    assert len(client.requests) == 1
    request = client.requests[0]
    assert [payload.filename for payload in request.media] == ["a.png", "b.webp", "z.JPG"]
    assert "same identity" in request.prompt
    assert result.sidecar_json is not None
    sidecar = json.loads(result.sidecar_json.read_text(encoding="utf-8"))
    assert sidecar["source_image"] == str(representative)
    assert sidecar["source_images"] == [
        str(group / "a.png"),
        str(group / "nested" / "b.webp"),
        str(group / "z.JPG"),
    ]
    assert sidecar["media"]["kind"] == "image_group"
    assert sidecar["media"]["num_views"] == 3
    assert sidecar["media"]["total_bytes"] == 16
    assert result.image_json is not None
    compact = json.loads(result.image_json.read_text(encoding="utf-8"))
    assert compact["num_views"] == 3
    assert compact["source_images"] == sidecar["source_images"]


def test_dense_captioner_evenly_caps_image_group(tmp_path: Path) -> None:
    group = tmp_path / "views"
    group.mkdir()
    for index in range(5):
        (group / f"{index}.jpg").write_bytes(str(index).encode())
    representative = group / "0.jpg"
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _StaticClient("A person viewed from multiple angles.")
    captioner = DenseCaptioner(
        config=CaptioningConfig(image_group_dir=str(group), max_group_images=3),
        vlm_client=client,
    )

    captioner.run(
        media_path=representative,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(representative),
    )

    assert [payload.filename for payload in client.requests[0].media] == [
        "0.jpg",
        "2.jpg",
        "4.jpg",
    ]


@pytest.mark.parametrize("create_directory", [False, True])
def test_dense_captioner_rejects_missing_or_empty_image_group(
    tmp_path: Path,
    create_directory: bool,
) -> None:
    group = tmp_path / "views"
    if create_directory:
        group.mkdir()
    representative = tmp_path / "representative.jpg"
    representative.write_bytes(b"representative")
    captioner = DenseCaptioner(
        config=CaptioningConfig(image_group_dir=str(group)),
        vlm_client=_StaticClient("unused"),
    )

    match = "contains no supported images" if create_directory else "does not exist"
    with pytest.raises(ValueError, match=match):
        captioner.run(
            media_path=representative,
            scene_paths=ensure_scene_skeleton(tmp_path / "scene"),
            scene_ctx=SceneContext.from_input(representative),
        )


def test_dense_captioner_writes_compact_video_sidecar_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_window_video(monkeypatch, frame_caption_payload=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video fixture")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _StaticClient("A forklift crosses the loading bay.")
    captioner = DenseCaptioner(
        config=CaptioningConfig(window_frames=0, window_seconds=5.0, media_mode="frames"),
        vlm_client=client,
    )

    result = captioner.run(
        media_path=video,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(video),
    )

    assert result.success is True
    assert result.sidecar_json is not None
    sidecar = json.loads(result.sidecar_json.read_text())
    window = sidecar["windows"][0]
    assert sidecar["windowing"]["caption_key"] == "description"
    assert "raw_caption_key" not in sidecar["windowing"]
    assert sidecar["token_counts"] == {
        "prompt_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert sidecar["finish_reasons"] == ["stop"]
    assert window["description"] == "A forklift crosses the loading bay."
    assert window["input_media_mode"] == "frames"
    assert "input_media" not in window
    assert "finish_reason" not in window
    assert "token_counts" not in window
    assert "caption_status" not in window
    assert "caption_failure_reason" not in window
    assert "call" not in window
    assert "sampled_frames" not in window
    assert "vlm_caption" not in window
    assert "parsed" not in window


def test_dense_captioner_can_preserve_raw_video_model_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_window_video(monkeypatch, frame_caption_payload=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video fixture")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _StaticClient('{"caption": "A worker enters the scene."}')
    captioner = DenseCaptioner(
        config=CaptioningConfig(
            window_frames=0,
            window_seconds=5.0,
            media_mode="frames",
            preserve_raw_model_output=True,
        ),
        vlm_client=client,
    )

    result = captioner.run(
        media_path=video,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(video),
    )

    assert result.sidecar_json is not None
    sidecar = json.loads(result.sidecar_json.read_text())
    window = sidecar["windows"][0]
    assert sidecar["windowing"]["caption_key"] == "description"
    assert sidecar["windowing"]["raw_caption_key"] == "vlm_caption"
    assert sidecar["windowing"]["parsed_key"] == "parsed"
    assert window["description"] == "A worker enters the scene."
    assert window["caption_status"] == "success"
    assert window["caption_failure_reason"] is None
    assert window["call"]["model"] == "test-model"
    assert window["token_counts"] == {
        "prompt_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert window["finish_reason"] == "stop"
    assert window["input_media"][0]["mime_type"] == "image/jpeg"
    assert "persisted" not in window["input_media"][0]
    assert window["vlm_caption"] == '{"caption": "A worker enters the scene."}'
    assert window["parsed"] == {"caption": "A worker enters the scene."}


def test_auto_media_mode_falls_back_when_endpoint_rejects_video_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_window_video(monkeypatch, frame_caption_payload=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video fixture")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _VideoRejectThenFrameClient()
    captioner = DenseCaptioner(
        config=CaptioningConfig(window_frames=0, window_seconds=5.0, media_mode="auto"),
        vlm_client=client,
    )

    result = captioner.run(
        media_path=video,
        scene_paths=scene_paths,
        scene_ctx=SceneContext.from_input(video),
    )

    assert result.sidecar_json is not None
    sidecar = json.loads(result.sidecar_json.read_text())
    window = sidecar["windows"][0]
    assert [request.media[0].mime_type for request in client.requests] == [
        "video/mp4",
        "image/jpeg",
    ]
    assert window["input_media_mode"] == "frames"
    assert "input_media" not in window
    assert window["media_failure_reason"] == "video_model_call_failed:RuntimeError"
    assert window["description"] == "Frame fallback caption."
    assert sidecar["token_counts"] == {
        "prompt_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
    }
    assert sidecar["finish_reasons"] == ["stop"]
    assert "finish_reason" not in window
    assert "token_counts" not in window
    assert "call" not in window


def test_auto_media_mode_reraises_unexpected_video_payload_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extract_window_video_payload(
        *,
        media_path: Path,
        window: CaptionWindow,
        total_frames: int,
    ) -> MediaPayload:
        raise RuntimeError("unexpected payload regression")

    monkeypatch.setattr(
        captioner_module,
        "extract_window_video_payload",
        fake_extract_window_video_payload,
    )
    captioner = DenseCaptioner(
        config=CaptioningConfig(media_mode="auto"),
        vlm_client=_StaticClient("{}"),
    )

    with pytest.raises(RuntimeError, match="unexpected payload regression"):
        captioner._window_media_payloads(
            media_path=tmp_path / "clip.mp4",
            window=CaptionWindow(
                index=0,
                start_s=0.0,
                end_s=1.0,
                start_frame=0,
                end_frame=9,
            ),
            info=VideoInfo(fps=10.0, frame_count=10, width=640, height=480),
        )


def test_sidecar_filename_rejects_path_like_values() -> None:
    assert _safe_sidecar_filename("metadata_chunk.json") == "metadata_chunk.json"

    for value in (
        "",
        "../metadata_chunk.json",
        "captioning/metadata_chunk.json",
        "captioning\\metadata_chunk.json",
        "/var/metadata_chunk.json",
        "metadata..json",
        "metadata chunk.json",
    ):
        with pytest.raises(ValueError, match="sidecar_filename must be a plain filename"):
            _safe_sidecar_filename(value)


def _patch_single_window_video(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_caption_payload: bool,
) -> None:
    def fake_probe_video(path: Path) -> VideoInfo:
        return VideoInfo(fps=10.0, frame_count=50, width=640, height=480)

    def fake_plan_windows(info: VideoInfo, config: CaptioningConfig) -> list[CaptionWindow]:
        return [
            CaptionWindow(
                index=0,
                start_s=0.0,
                end_s=5.0,
                start_frame=0,
                end_frame=49,
            )
        ]

    def fake_extract_window_video_payload(
        *,
        media_path: Path,
        window: CaptionWindow,
        total_frames: int,
    ) -> MediaPayload:
        return MediaPayload(
            mime_type="video/mp4",
            data_base64="dmlydHVhbC12aWRlbw==",
            filename="clip_window_000.mp4",
            frame_index=window.start_frame,
            time_s=window.start_s,
            num_bytes=13,
        )

    def fake_extract_window_frames(
        *,
        media_path: Path,
        window: CaptionWindow,
        source_fps: float,
        sampling_fps: float,
        max_frames: int,
        resolution: int,
    ) -> list[MediaPayload]:
        return [
            MediaPayload(
                mime_type="image/jpeg",
                data_base64="dmlydHVhbC1qcGVn",
                filename="clip_000000.jpg",
                frame_index=window.start_frame,
                time_s=window.start_s,
                width=640,
                height=480,
                num_bytes=12,
            )
        ]

    monkeypatch.setattr(captioner_module, "probe_video", fake_probe_video)
    monkeypatch.setattr(captioner_module, "plan_windows", fake_plan_windows)
    monkeypatch.setattr(
        captioner_module,
        "extract_window_video_payload",
        fake_extract_window_video_payload,
    )
    if frame_caption_payload:
        monkeypatch.setattr(captioner_module, "extract_window_frames", fake_extract_window_frames)
