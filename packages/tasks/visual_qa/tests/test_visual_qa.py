# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from core import DataEntry, ScenePipelineState, read_pipeline_state, write_pipeline_state
from visual_qa import VisualQaConfig, VisualQaTask
from visual_qa import task as vqa_task
from visual_qa.artifacts import VISUAL_QA_ARTIFACTS_KEY, VisualQaArtifactsState
from visual_qa.bank import (
    aggregate_window_items,
    load_question_bank,
    normalize_answer,
    normalize_items,
)
from visual_qa.clients import ChatRequest
from visual_qa.media import load_crop_payloads, load_crop_payloads_from_files
from visual_qa.sidecar import extract_window_item_groups, resolve_sidecar, write_json_object


class _StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[ChatRequest] = []
        self.last_call_metadata: dict[str, object] = {
            "provider": "openai-compatible",
            "model": "test-model",
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


class _SequenceClient(_StaticClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    def generate(self, request: ChatRequest) -> str:
        self.requests.append(request)
        return next(self.responses)


def test_generation_clients_omit_api_key_env_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _StaticClient:
        calls.append(kwargs)
        return _StaticClient("{}")

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)

    VisualQaTask(
        VisualQaConfig(
            generation_mode="window-vlm-llm",
            question_bank_file="bank.json",
            vlm_model="vlm-test",
            llm_model="llm-test",
            vlm_endpoint_url="http://vlm.test/v1",
            llm_endpoint_url="http://llm.test/v1",
        )
    )

    assert calls[0]["model"] == "vlm-test"
    assert "api_key_env" not in calls[0]
    assert "fallback_api_key_envs" not in calls[0]
    assert calls[1]["model"] == "llm-test"
    assert "api_key_env" not in calls[1]
    assert "fallback_api_key_envs" not in calls[1]


def test_normalizes_source_metadata_and_aggregates_to_sidecar(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    (scene_dir / "sidecars").mkdir(parents=True)

    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "aggregation": "any",
                },
                {
                    "id": "incident_type",
                    "question": "What type of incident is visible?",
                    "options": ["A. fall", "B. collision"],
                    "include_if": {"incident": "Yes"},
                },
            ]
        },
    )
    _write_json(
        scene_dir / "sidecars" / "metadata.json",
        {
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "llm_enhanced_caption": _fenced(
                        {
                            "mcq": [
                                {
                                    "id": "incident",
                                    "question": "Is there an incident?",
                                    "options": ["Yes", "No"],
                                    "answer": "No",
                                }
                            ]
                        }
                    ),
                },
                {
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "llm_enhanced_caption": _fenced(
                        {
                            "mcq": [
                                {
                                    "id": "incident",
                                    "question": "Is there an incident?",
                                    "options": ["Yes", "No"],
                                    "answer": "Yes",
                                },
                                {
                                    "id": "incident_type",
                                    "question": "What type of incident is visible?",
                                    "options": ["A. fall", "B. collision"],
                                    "answer": "A",
                                    "reasoning_trace": "The person falls in the roadway.",
                                },
                            ]
                        }
                    ),
                },
            ]
        },
    )

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path)))
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    assert [item["id"] for item in items["items"]] == ["incident", "incident_type"]
    assert items["items"][0]["answer"] == "Yes"
    assert items["items"][1]["answer"] == "A. fall"
    mcq = json.loads((scene_dir / "task" / "mcq.json").read_text(encoding="utf-8"))
    bcq = json.loads((scene_dir / "task" / "bcq.json").read_text(encoding="utf-8"))
    assert [item["video_id"] for item in mcq["items"]] == ["clip"]
    assert mcq["items"][0]["question"] == "What type of incident is visible?"
    assert mcq["items"][0]["answer"] == "A"
    assert bcq["items"][0]["video_id"] == "clip"
    assert bcq["items"][0]["answer"] == "Yes"

    state = read_pipeline_state(scene_dir)
    visual_qa_artifacts = VisualQaArtifactsState.model_validate(
        state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY]
    )
    assert visual_qa_artifacts.success is True
    assert visual_qa_artifacts.items_json is not None
    assert state.annotation_export is not None
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert outcomes["mcq"].success is True
    assert outcomes["bcq"].success is True
    assert outcomes["open_qa"].success is False


def _seed_normalize_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Seed a minimal normalize-only visual_qa scene; return media/scene/bank."""
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    (scene_dir / "sidecars").mkdir(parents=True)
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "aggregation": "any",
                }
            ]
        },
    )
    _write_json(
        scene_dir / "sidecars" / "metadata.json",
        {
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "llm_enhanced_caption": _fenced(
                        {
                            "mcq": [
                                {
                                    "id": "incident",
                                    "question": "Is there an incident?",
                                    "options": ["Yes", "No"],
                                    "answer": "Yes",
                                }
                            ]
                        }
                    ),
                }
            ]
        },
    )
    return media_path, scene_dir, bank_path


def test_state_artifacts_key_namespaces_and_mirrors_canonical(tmp_path: Path) -> None:
    media_path, scene_dir, bank_path = _seed_normalize_scene(tmp_path)

    task = VisualQaTask(
        VisualQaConfig(
            question_bank_file=str(bank_path),
            state_artifacts_key="visual_qa_anomaly_search",
        )
    )
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    state = read_pipeline_state(scene_dir)
    # Namespaced slice recorded under the configured key.
    assert "visual_qa_anomaly_search" in state.task_artifacts
    namespaced = VisualQaArtifactsState.model_validate(
        state.task_artifacts["visual_qa_anomaly_search"]
    )
    assert namespaced.success is True
    assert namespaced.items_json is not None
    # Canonical "latest" pointer also written, tagged with the variant.
    assert state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY]["variant"] == "visual_qa_anomaly_search"


def test_state_artifacts_key_default_writes_only_canonical(tmp_path: Path) -> None:
    media_path, scene_dir, bank_path = _seed_normalize_scene(tmp_path)

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path)))
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    state = read_pipeline_state(scene_dir)
    assert VISUAL_QA_ARTIFACTS_KEY in state.task_artifacts
    # No variant tag and no extra namespaced key when default is used.
    assert "variant" not in state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY]
    assert list(state.task_artifacts) == [VISUAL_QA_ARTIFACTS_KEY]


def test_multi_pass_state_artifacts_keys_do_not_clobber(tmp_path: Path) -> None:
    media_path, scene_dir, bank_path = _seed_normalize_scene(tmp_path)
    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    VisualQaTask(
        VisualQaConfig(
            question_bank_file=str(bank_path),
            state_artifacts_key="visual_qa_anomaly",
        )
    ).run(entry)
    VisualQaTask(
        VisualQaConfig(
            question_bank_file=str(bank_path),
            state_artifacts_key="visual_qa_anomaly_search",
        )
    ).run(entry)

    state = read_pipeline_state(scene_dir)
    # Both passes' provenance survives; canonical points at the most recent run.
    assert "visual_qa_anomaly" in state.task_artifacts
    assert "visual_qa_anomaly_search" in state.task_artifacts
    assert state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY]["variant"] == "visual_qa_anomaly_search"


def test_emit_flat_qa_tasks_false_skips_flat_task_files(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    (scene_dir / "sidecars").mkdir(parents=True)

    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "aggregation": "any",
                }
            ]
        },
    )
    _write_json(
        scene_dir / "sidecars" / "metadata.json",
        {
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "llm_enhanced_caption": _fenced(
                        {
                            "mcq": [
                                {
                                    "id": "incident",
                                    "question": "Is there an incident?",
                                    "options": ["Yes", "No"],
                                    "answer": "Yes",
                                }
                            ]
                        }
                    ),
                }
            ]
        },
    )

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path), emit_flat_qa_tasks=False))
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    # The normalized sidecar is still produced; only the flat DAFT task files are skipped.
    assert (scene_dir / "sidecars" / "visual_qa" / "items.json").exists()
    for flat_name in ("mcq.json", "bcq.json", "open_qa.json"):
        assert not (scene_dir / "task" / flat_name).exists()


def test_normalize_uses_scene_directory_media_id_for_pipeline_active_media(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "clip_001"
    active = scene_dir / "sidecars" / "active.jpg"
    active.parent.mkdir(parents=True)
    active.touch()
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "object_visible",
                    "question": "Is an object visible?",
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
                    "window_index": 0,
                    "items": [
                        {
                            "id": "object_visible",
                            "question": "Is an object visible?",
                            "options": ["Yes", "No"],
                            "answer": "Yes",
                        }
                    ],
                }
            ]
        },
    )

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path)))
    task.run(DataEntry(media_path=str(active), data_path=str(scene_dir)))

    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    windows = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.normalized.json").read_text(
            encoding="utf-8"
        )
    )
    assert items["media_id"] == "clip_001"
    assert windows["media_id"] == "clip_001"


def test_normalize_uses_persisted_media_id_for_staged_pipeline_active_media(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "pipeline_data_tmp"
    active = scene_dir / "sidecars" / "active.jpg"
    active.parent.mkdir(parents=True)
    active.touch()
    write_pipeline_state(scene_dir, ScenePipelineState(media_id="clip_001"))
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "object_visible",
                    "question": "Is an object visible?",
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
                    "window_index": 0,
                    "items": [
                        {
                            "id": "object_visible",
                            "question": "Is an object visible?",
                            "options": ["Yes", "No"],
                            "answer": "Yes",
                        }
                    ],
                }
            ]
        },
    )

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path)))
    task.run(DataEntry(media_path=str(active), data_path=str(scene_dir)))

    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    windows = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.normalized.json").read_text(
            encoding="utf-8"
        )
    )
    assert items["media_id"] == "clip_001"
    assert windows["media_id"] == "clip_001"


def test_window_direct_vlm_generates_and_normalizes_image_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-a-real-jpeg-but-enough-for-payload")
    scene_dir = tmp_path / "scene"
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "road_occupied",
                    "question": "Is the road occupied?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    vlm_client = _StaticClient(
        _fenced(
            {
                "items": [
                    {
                        "id": "road_occupied",
                        "question": "Is the road occupied?",
                        "options": ["Yes", "No"],
                        "answer": "A",
                    }
                ]
            }
        )
    )

    def factory(**_kwargs: object) -> _StaticClient:
        return vlm_client

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            vlm_model="vlm-test",
        )
    )
    task.run(DataEntry(media_path=str(image_path), data_path=str(scene_dir)))

    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    assert raw["generation_mode"] == "window-direct-vlm"
    assert raw["windows"][0]["visual_qa_generation_mode"] == "window-direct-vlm"
    assert items["items"][0]["answer"] == "Yes"
    assert len(vlm_client.requests) == 1
    assert vlm_client.requests[0].media
    assert "QUESTION BANK" in vlm_client.requests[0].prompt


def test_window_direct_vlm_sends_recursive_image_group_as_one_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "views"
    (group / "nested").mkdir(parents=True)
    (group / "z.JPG").write_bytes(b"third")
    (group / "a.png").write_bytes(b"first")
    (group / "nested" / "b.webp").write_bytes(b"second")
    (group / "ignore.txt").write_text("not media", encoding="utf-8")
    representative = group / "a.png"
    scene_dir = tmp_path / "scene"
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {"questions": [{"id": "shirt", "question": "Shirt color?", "options": []}]},
    )
    vlm_client = _StaticClient(_fenced({"items": [{"id": "shirt", "answer": "blue"}]}))
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", lambda **_kwargs: vlm_client)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            image_group_dir=str(group),
            vlm_model="vlm-test",
        )
    )
    task.run(DataEntry(media_path=str(representative), data_path=str(scene_dir)))

    assert len(vlm_client.requests) == 1
    assert [payload.filename for payload in vlm_client.requests[0].media] == [
        "a.png",
        "b.webp",
        "z.JPG",
    ]
    assert "same identity" in vlm_client.requests[0].prompt
    source_images = [
        str(group / "a.png"),
        str(group / "nested" / "b.webp"),
        str(group / "z.JPG"),
    ]
    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    normalized = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.normalized.json").read_text(
            encoding="utf-8"
        )
    )
    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    assert raw["source_image_group"] == str(group)
    assert raw["windows"][0]["source_images"] == source_images
    assert raw["windows"][0]["num_views"] == 3
    assert normalized["windows"][0]["source_images"] == source_images
    assert normalized["windows"][0]["num_views"] == 3
    assert items["items"][0]["answer"] == "blue"


def test_window_direct_vlm_evenly_caps_image_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "views"
    group.mkdir()
    for index in range(5):
        (group / f"{index}.jpg").write_bytes(str(index).encode())
    bank_path = tmp_path / "question_bank.json"
    _write_json(bank_path, {"questions": [{"id": "q", "question": "Q?", "options": []}]})
    vlm_client = _StaticClient(_fenced({"items": [{"id": "q", "answer": "answer"}]}))
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", lambda **_kwargs: vlm_client)

    VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            image_group_dir=str(group),
            max_group_images=3,
        )
    ).run(DataEntry(media_path=str(group / "0.jpg"), data_path=str(tmp_path / "scene")))

    assert [payload.filename for payload in vlm_client.requests[0].media] == [
        "0.jpg",
        "2.jpg",
        "4.jpg",
    ]


@pytest.mark.parametrize("create_directory", [False, True])
def test_image_group_rejects_missing_or_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_directory: bool,
) -> None:
    group = tmp_path / "views"
    if create_directory:
        group.mkdir()
    representative = tmp_path / "representative.jpg"
    representative.write_bytes(b"representative")
    bank_path = tmp_path / "question_bank.json"
    _write_json(bank_path, {"questions": []})
    monkeypatch.setattr(
        "visual_qa.task.create_endpoint_client", lambda **_kwargs: _StaticClient("unused")
    )
    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            image_group_dir=str(group),
        )
    )

    match = "contains no supported images" if create_directory else "does not exist"
    with pytest.raises(ValueError, match=match):
        task.run(DataEntry(media_path=str(representative), data_path=str(tmp_path / "scene")))


def test_remote_question_bank_file_is_downloaded_for_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-a-real-jpeg-but-enough-for-payload")
    scene_dir = tmp_path / "scene"
    bank_path = tmp_path / "downloaded_question_bank.json"
    remote_bank_path = "s3://bucket/question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "road_occupied",
                    "question": "Is the road occupied?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    vlm_client = _StaticClient(
        _fenced(
            {
                "items": [
                    {
                        "id": "road_occupied",
                        "question": "Is the road occupied?",
                        "options": ["Yes", "No"],
                        "answer": "A",
                    }
                ]
            }
        )
    )
    downloads: list[tuple[str, str | None, bool]] = []

    def download_if_remote(
        _storage: object,
        path: str,
        local_path: str | None = None,
        *,
        is_file: bool,
    ) -> str:
        downloads.append((path, local_path, is_file))
        return str(bank_path)

    def factory(**_kwargs: object) -> _StaticClient:
        return vlm_client

    monkeypatch.setattr("visual_qa.task.MSCStorage.download_if_remote", download_if_remote)
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=remote_bank_path,
            vlm_model="vlm-test",
        )
    )
    task.run(DataEntry(media_path=str(image_path), data_path=str(scene_dir)))

    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )

    assert downloads == [(remote_bank_path, None, True)]
    assert raw["question_bank"]["path"] == remote_bank_path
    assert items["question_bank"]["path"] == remote_bank_path
    assert items["items"][0]["answer"] == "Yes"
    assert "road_occupied" in vlm_client.requests[0].prompt


def test_window_vlm_llm_generates_evidence_then_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-a-real-jpeg-but-enough-for-payload")
    scene_dir = tmp_path / "scene"
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "truck_visible",
                    "question": "Is a truck visible?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    vlm_client = _StaticClient(_fenced({"caption": "A truck is parked on the road."}))
    llm_client = _StaticClient(
        _fenced(
            {
                "items": [
                    {
                        "id": "truck_visible",
                        "question": "Is a truck visible?",
                        "options": ["Yes", "No"],
                        "answer": "Yes",
                    }
                ]
            }
        )
    )
    clients = {"vlm-test": vlm_client, "llm-test": llm_client}

    def factory(*, model: str, **_kwargs: object) -> _StaticClient:
        return clients[model]

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-vlm-llm",
            question_bank_file=str(bank_path),
            vlm_model="vlm-test",
            llm_model="llm-test",
            evidence_prompt_text="Extract visual evidence.",
        )
    )
    task.run(DataEntry(media_path=str(image_path), data_path=str(scene_dir)))

    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    assert raw["windows"][0]["visual_evidence"] == "A truck is parked on the road."
    assert raw["windows"][0]["items"][0]["answer"] == "Yes"
    assert len(vlm_client.requests) == 1
    assert len(llm_client.requests) == 1
    assert vlm_client.requests[0].media
    assert llm_client.requests[0].media == ()
    assert "WINDOW EVIDENCE" in llm_client.requests[0].prompt
    assert "A truck is parked on the road." in llm_client.requests[0].prompt


def test_question_driven_mode_generates_legacy_prompts_then_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"image")
    scene_dir = tmp_path / "scene"
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "collision",
                    "question": "Is there a collision?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    generated_scene_prompt = (
        "## SCENE SUMMARY\n## GLOBAL CONTEXT\n## PER-QUESTION EVIDENCE\n"
        "### collision Is there a collision?"
    )
    vlm_client = _StaticClient("A sedan has collided with another vehicle.")
    llm_client = _SequenceClient(
        [
            generated_scene_prompt,
            _fenced(
                {
                    "mcq": [
                        {
                            "id": "collision",
                            "question": "Is there a collision?",
                            "options": ["Yes", "No"],
                            "answer": "Yes",
                        }
                    ]
                }
            ),
        ]
    )
    clients = {"vlm-test": vlm_client, "llm-test": llm_client}

    def factory(*, model: str, **_kwargs: object) -> _StaticClient:
        return clients[model]

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)
    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="question-driven-vlm-llm",
            question_bank_file=str(bank_path),
            vlm_model="vlm-test",
            llm_model="llm-test",
            temperature=0.0,
        )
    )
    monkeypatch.setattr("visual_qa.task.read_image_payload", lambda _path: ())
    task.run(DataEntry(media_path=str(image_path), data_path=str(scene_dir)))

    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    assert raw["generation_mode"] == "question-driven-vlm-llm"
    assert raw["windows"][0]["items"][0]["answer"] == "Yes"
    assert len(llm_client.requests) == 2
    assert llm_client.requests[0].temperature == 0.0
    assert '"id": "collision"' in llm_client.requests[0].prompt
    assert llm_client.requests[1].system_prompt is not None
    assert "QUESTION BANK (AUTHORITATIVE)" in llm_client.requests[1].system_prompt
    assert llm_client.requests[1].prompt == "A sedan has collided with another vehicle."
    prompts_dir = scene_dir / "sidecars" / "visual_qa" / "prompts"
    assert (prompts_dir / "scene_prompt.used.md").is_file()
    assert (prompts_dir / "mcq_prompt.used.md").is_file()


def test_metadata_llm_generates_qa_from_captioning_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "worker_visible",
                    "question": "Is a worker visible?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    _write_json(
        scene_dir / "sidecars" / "captioning" / "metadata_chunk.json",
        {
            "windows": [
                {
                    "index": 0,
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A worker stands beside the vehicle.",
                }
            ]
        },
    )
    llm_client = _StaticClient(
        _fenced(
            {
                "items": [
                    {
                        "id": "worker_visible",
                        "question": "Is a worker visible?",
                        "options": ["Yes", "No"],
                        "answer": "Yes",
                    }
                ]
            }
        )
    )
    factory_models: list[str] = []

    def factory(*, model: str, **_kwargs: object) -> _StaticClient:
        factory_models.append(model)
        return llm_client

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", factory)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="metadata-llm",
            question_bank_file=str(bank_path),
            llm_model="llm-test",
        )
    )
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    raw = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.json").read_text(encoding="utf-8")
    )
    items = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "items.json").read_text(encoding="utf-8")
    )
    assert factory_models == ["llm-test"]
    assert raw["generation_mode"] == "metadata-llm"
    assert raw["source_sidecar"].endswith("sidecars/captioning/metadata_chunk.json")
    assert items["items"][0]["answer"] == "Yes"
    assert "A worker stands beside the vehicle." in llm_client.requests[0].prompt


def test_generation_modes_require_question_bank() -> None:
    with pytest.raises(ValueError, match="generation_mode=window-direct-vlm requires"):
        VisualQaConfig(generation_mode="window-direct-vlm")


def test_include_if_drops_dependent_question_when_gate_fails(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {"id": "gate", "question": "Is event visible?", "options": ["Yes", "No"]},
                {
                    "id": "detail",
                    "question": "Describe the event.",
                    "options": [],
                    "include_if": {"gate": "Yes"},
                },
            ]
        },
    )
    bank = load_question_bank(bank_path)
    normalized, errors = normalize_items(
        [
            {
                "id": "gate",
                "question": "Is event visible?",
                "options": ["Yes", "No"],
                "answer": "No",
            },
            {"id": "detail", "question": "Describe the event.", "options": [], "answer": "fall"},
        ],
        bank=bank,
        strict_answers=True,
        open_ended=False,
    )

    payload = aggregate_window_items([normalized], bank=bank, media_id="clip", aggregate=True)
    assert errors == []
    assert [item["id"] for item in payload["items"]] == ["gate"]


def test_load_question_bank_rejects_invalid_entries(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.json"
    _write_json(bank_path, {"questions": ["not an object"]})

    with pytest.raises(ValueError, match="question bank entry 0 must be an object"):
        load_question_bank(bank_path)


def test_load_question_bank_rejects_duplicate_ids(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {"id": "incident", "question": "Is there an incident?"},
                {"id": "incident", "question": "Is the incident severe?"},
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate id 'incident' at entry 1"):
        load_question_bank(bank_path)


def test_aggregated_reasoning_trace_matches_selected_answer(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.json"
    _write_json(
        bank_path,
        {
            "questions": [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                }
            ]
        },
    )
    bank = load_question_bank(bank_path)

    payload = aggregate_window_items(
        [
            [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "answer": "No",
                    "reasoning_trace": "No issue in the first window.",
                }
            ],
            [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "answer": "Yes",
                    "reasoning_trace": "A worker falls in the second window.",
                }
            ],
            [
                {
                    "id": "incident",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "answer": "Yes",
                    "reasoning_trace": "The fall continues in the third window.",
                }
            ],
        ],
        bank=bank,
        media_id="clip",
        aggregate=True,
    )

    assert payload["items"][0]["answer"] == "Yes"
    assert payload["items"][0]["reasoning_trace"] == "A worker falls in the second window."


def test_task_omits_items_state_when_empty_marker_disabled(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    bank_path = tmp_path / "bank.json"
    bank_payload = {
        "questions": [
            {"id": "gate", "question": "Is event visible?", "options": ["Yes", "No"]},
            {
                "id": "detail",
                "question": "Describe the event.",
                "options": [],
                "include_if": {"gate": "Yes"},
            },
        ]
    }
    raw_items = [
        {"id": "detail", "question": "Describe the event.", "options": [], "answer": "fall"},
    ]
    _write_json(bank_path, bank_payload)
    _write_json(
        scene_dir / "sidecars" / "visual_qa" / "windows.json", {"windows": [{"items": raw_items}]}
    )

    bank = load_question_bank(bank_path)
    normalized, errors = normalize_items(
        raw_items,
        bank=bank,
        strict_answers=True,
        open_ended=False,
    )
    payload = aggregate_window_items([normalized], bank=bank, media_id="clip", aggregate=True)
    assert errors == []
    assert payload["items"] == []

    task = VisualQaTask(VisualQaConfig(question_bank_file=str(bank_path), write_empty_marker=False))
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    state = read_pipeline_state(scene_dir)
    visual_qa_artifacts = VisualQaArtifactsState.model_validate(
        state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY]
    )
    assert visual_qa_artifacts.success is True
    assert visual_qa_artifacts.items_json is None
    assert visual_qa_artifacts.windows_json is not None
    assert not (scene_dir / "sidecars" / "visual_qa" / "items.json").exists()


def test_task_preserves_empty_window_item_alignment(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "sidecars" / "visual_qa" / "windows.json",
        {
            "windows": [
                {"window_index": 0, "items": []},
                {
                    "window_index": 1,
                    "items": [
                        {
                            "id": "incident",
                            "question": "Is there an incident?",
                            "options": ["Yes", "No"],
                            "answer": "Yes",
                        }
                    ],
                },
            ]
        },
    )

    task = VisualQaTask()
    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    windows = json.loads(
        (scene_dir / "sidecars" / "visual_qa" / "windows.normalized.json").read_text(
            encoding="utf-8"
        )
    )
    assert windows["windows"][0]["window_index"] == 0
    assert windows["windows"][0]["items"] == []
    assert windows["windows"][1]["window_index"] == 1
    assert windows["windows"][1]["items"][0]["id"] == "incident"


def test_lowercase_letter_answers_map_to_options() -> None:
    assert normalize_answer("a", ["A. fall", "B. collision"]) == "A. fall"


def test_lowercase_letter_answers_map_to_extended_options() -> None:
    options = [f"option-{idx}" for idx in range(28)]

    assert normalize_answer("a", options) == "option-26"
    assert normalize_answer("A", options) == "option-0"


def test_window_metadata_preserves_source_window_index() -> None:
    _groups, windows = extract_window_item_groups(
        {
            "windows": [
                {"window_index": 42, "items": []},
                {"index": 7.0, "items": []},
                {"window_index": "bad", "index": 9, "items": []},
                {"items": []},
            ]
        }
    )

    assert [window["window_index"] for window in windows] == [42, 7, 9, 3]


def test_window_metadata_excludes_bool_numeric_fields() -> None:
    _groups, windows = extract_window_item_groups(
        {
            "windows": [
                {
                    "window_index": 0,
                    "start_s": False,
                    "end_s": True,
                    "start_frame": 1,
                    "end_frame": 2.0,
                    "items": [],
                }
            ]
        }
    )

    assert "start_s" not in windows[0]
    assert "end_s" not in windows[0]
    assert windows[0]["start_frame"] == 1
    assert windows[0]["end_frame"] == 2.0


def test_write_json_object_uses_unique_temp_name(tmp_path: Path) -> None:
    target = tmp_path / "items.json"
    stale_fixed_tmp = tmp_path / "items.json.tmp"
    stale_fixed_tmp.write_text("stale", encoding="utf-8")

    write_json_object(target, {"items": []})

    assert json.loads(target.read_text(encoding="utf-8")) == {"items": []}
    assert stale_fixed_tmp.read_text(encoding="utf-8") == "stale"
    assert list(tmp_path.glob("items.json.*.tmp")) == []


@pytest.mark.parametrize("relative_name", ["../outside.json", "/etc/passwd"])
def test_resolve_sidecar_rejects_paths_outside_scene_sidecars(
    tmp_path: Path, relative_name: str
) -> None:
    root = tmp_path / "sidecars"

    with pytest.raises(ValueError, match="sidecar path escapes scene sidecars root"):
        resolve_sidecar(root, relative_name)


def test_load_crop_payloads_samples_endpoints(tmp_path: Path) -> None:
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    for i in range(10):
        (crop_dir / f"crop_{i:02d}.jpg").write_bytes(b"fake-jpeg-bytes")

    payloads = load_crop_payloads(crop_dir, max_crops=8)
    names = [p.filename for p in payloads]

    assert len(names) == 8
    # Endpoint-inclusive sampling keeps both the first and last crop.
    assert names[0] == "crop_00.jpg"
    assert names[-1] == "crop_09.jpg"


def test_load_crop_payloads_from_files_ignores_unlisted_and_missing(tmp_path: Path) -> None:
    """Only the listed, on-disk crop files are loaded; stale dir files are ignored."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    listed = []
    for i in range(2):
        path = crop_dir / f"crop_{i:02d}.jpg"
        path.write_bytes(b"fake-jpeg-bytes")
        listed.append(path)
    # A stale crop left in the directory by an earlier run must not leak in.
    (crop_dir / "crop_99.jpg").write_bytes(b"stale")
    # A recorded-but-deleted crop must be skipped, not error.
    listed.append(crop_dir / "crop_missing.jpg")

    payloads = load_crop_payloads_from_files(listed, max_crops=8)
    names = [p.filename for p in payloads]

    assert names == ["crop_00.jpg", "crop_01.jpg"]


def test_track_crops_prefers_recorded_crops_over_stale_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded ``crops`` win over a directory that also holds stale crop files."""
    scene_dir = tmp_path / "scene"
    sidecars = scene_dir / "sidecars"
    crop_dir = sidecars / "detection_and_tracking/crops/track_0001"
    crop_dir.mkdir(parents=True, exist_ok=True)
    recorded = [f"detection_and_tracking/crops/track_0001/crop_{i:02d}.jpg" for i in range(2)]
    for rel in recorded:
        (sidecars / rel).write_bytes(b"fake-jpeg-bytes")
    # Stale crops from a previous, longer run still sit in the directory.
    for i in range(2, 6):
        (crop_dir / f"crop_{i:02d}.jpg").write_bytes(b"stale")
    _write_json(
        sidecars / "detection_and_tracking/tracks.json",
        {
            "crop_root": "detection_and_tracking/crops",
            "tracks": [
                {
                    "track_id": 1,
                    "crop_dir": "detection_and_tracking/crops/track_0001",
                    "crops": recorded,
                }
            ],
        },
    )
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {"questions": [{"id": "gender", "question": "Gender?", "options": ["male", "female"]}]},
    )
    vlm_client = _StaticClient(_fenced({"items": [{"id": "gender", "answer": "male"}]}))
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", lambda **_kwargs: vlm_client)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            track_crops_sidecar="detection_and_tracking/tracks.json",
            max_crops_per_track=8,
            vlm_model="vlm-test",
        )
    )
    task.run(DataEntry(media_path=str(tmp_path / "chunk.mp4"), data_path=str(scene_dir)))

    raw = json.loads((sidecars / "visual_qa" / "windows.json").read_text(encoding="utf-8"))
    by_track_window = {w["track_id"]: w for w in raw["windows"]}
    # Exactly the 2 recorded crops are used, not the 6 files present on disk.
    assert by_track_window[1]["n_crops_in_chunk"] == 2
    assert len(vlm_client.requests) == 1
    assert len(vlm_client.requests[0].media) == 2


def test_track_crops_generation_emits_one_window_per_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_dir = tmp_path / "scene"
    sidecars = scene_dir / "sidecars"
    # Track 1 has more crops than max_crops_per_track (8) to exercise truncation;
    # track 2 stays under the cap.
    crop_counts = {1: 10, 2: 3}
    for track_id, count in crop_counts.items():
        crop_dir = sidecars / f"detection_and_tracking/crops/track_{track_id:04d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (crop_dir / f"crop_{i:02d}.jpg").write_bytes(b"fake-jpeg-bytes")
    _write_json(
        sidecars / "detection_and_tracking/tracks.json",
        {
            "crop_root": "detection_and_tracking/crops",
            "tracks": [
                {"track_id": 1, "crop_dir": "detection_and_tracking/crops/track_0001"},
                {"track_id": 2, "crop_dir": "detection_and_tracking/crops/track_0002"},
            ],
        },
    )
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {"questions": [{"id": "gender", "question": "Gender?", "options": ["male", "female"]}]},
    )
    vlm_client = _StaticClient(_fenced({"items": [{"id": "gender", "answer": "male"}]}))
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", lambda **_kwargs: vlm_client)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            track_crops_sidecar="detection_and_tracking/tracks.json",
            max_crops_per_track=8,
            vlm_model="vlm-test",
        )
    )
    task.run(DataEntry(media_path=str(tmp_path / "chunk.mp4"), data_path=str(scene_dir)))

    raw = json.loads((sidecars / "visual_qa" / "windows.json").read_text(encoding="utf-8"))
    normalized = json.loads(
        (sidecars / "visual_qa" / "windows.normalized.json").read_text(encoding="utf-8")
    )
    assert raw["crop_root"] == "detection_and_tracking/crops"
    assert [w["track_id"] for w in raw["windows"]] == [1, 2]
    by_track_window = {w["track_id"]: w for w in raw["windows"]}
    # Track 1's 10 crops are capped at max_crops_per_track=8; track 2 keeps 3.
    assert by_track_window[1]["n_crops_in_chunk"] == 8
    assert by_track_window[2]["n_crops_in_chunk"] == 3
    # One VLM call per track, each carrying that track's (capped) crops.
    assert len(vlm_client.requests) == 2
    assert len(vlm_client.requests[0].media) == 8
    assert len(vlm_client.requests[1].media) == 3
    # Normalized windows keep track identity and per-track items.
    by_track = {w["track_id"]: w for w in normalized["windows"]}
    assert set(by_track) == {1, 2}
    assert by_track[1]["items"][0]["answer"] == "male"


def test_track_crops_missing_sidecar_skips_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing track-crops sidecar (person-less scene) must not abort the run.

    Detection writes no ``tracks.json`` when it keeps zero tracks (an empty
    scene, or all tracks pruned by a crop gate). Visual QA must treat that as a
    zero-result scene and emit empty windows instead of raising, so one empty
    clip cannot fail an entire batch.
    """
    scene_dir = tmp_path / "scene"
    sidecars = scene_dir / "sidecars"
    sidecars.mkdir(parents=True, exist_ok=True)
    # Intentionally do NOT write detection_and_tracking/tracks.json.
    bank_path = tmp_path / "question_bank.json"
    _write_json(
        bank_path,
        {"questions": [{"id": "gender", "question": "Gender?", "options": ["male", "female"]}]},
    )
    vlm_client = _StaticClient(_fenced({"items": [{"id": "gender", "answer": "male"}]}))
    monkeypatch.setattr("visual_qa.task.create_endpoint_client", lambda **_kwargs: vlm_client)

    task = VisualQaTask(
        VisualQaConfig(
            generation_mode="window-direct-vlm",
            question_bank_file=str(bank_path),
            track_crops_sidecar="detection_and_tracking/tracks.json",
            max_crops_per_track=8,
            vlm_model="vlm-test",
        )
    )
    # Must not raise.
    task.run(DataEntry(media_path=str(tmp_path / "chunk.mp4"), data_path=str(scene_dir)))

    raw = json.loads((sidecars / "visual_qa" / "windows.json").read_text(encoding="utf-8"))
    assert raw["windows"] == []
    # No VLM call should happen when there are no tracks.
    assert vlm_client.requests == []


def test_track_crops_requires_vlm_window_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="track_crops_sidecar requires generation_mode"):
        VisualQaConfig(
            generation_mode="metadata-llm",
            question_bank_file=str(tmp_path / "bank.json"),
            track_crops_sidecar="detection_and_tracking/tracks.json",
        )


def test_track_crops_accepts_question_driven_mode(tmp_path: Path) -> None:
    config = VisualQaConfig(
        generation_mode="question-driven-vlm-llm",
        question_bank_file=str(tmp_path / "bank.json"),
        track_crops_sidecar="detection_and_tracking/tracks.json",
    )
    assert config.generation_mode == "question-driven-vlm-llm"
    assert config.track_crops_sidecar == "detection_and_tracking/tracks.json"


def test_image_group_dir_accepts_question_driven_mode(tmp_path: Path) -> None:
    config = VisualQaConfig(
        generation_mode="question-driven-vlm-llm",
        question_bank_file=str(tmp_path / "bank.json"),
        image_group_dir=str(tmp_path / "group"),
    )
    assert config.generation_mode == "question-driven-vlm-llm"
    assert config.image_group_dir == str(tmp_path / "group")


def test_image_group_dir_rejects_metadata_llm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_group_dir requires generation_mode"):
        VisualQaConfig(
            generation_mode="metadata-llm",
            question_bank_file=str(tmp_path / "bank.json"),
            image_group_dir=str(tmp_path / "group"),
        )


def test_parser_defaults_to_instruct() -> None:
    assert VisualQaConfig().parser == "instruct"


def test_parser_forwarded_to_vlm_and_llm_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> _StaticClient:
        captured.append(kwargs)
        return _StaticClient("{}")

    monkeypatch.setattr("visual_qa.task.create_endpoint_client", fake_factory)

    config = VisualQaConfig(parser="reasoning")
    vqa_task._build_vlm_client(config)
    vqa_task._build_llm_client(config)

    assert [call["parser"] for call in captured] == ["reasoning", "reasoning"]


def _fenced(payload: object) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
