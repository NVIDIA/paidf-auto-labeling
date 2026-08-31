# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chunk-level 3-bucket query generation (query_generation.py parity)."""

from __future__ import annotations

import json
from pathlib import Path

from core import ensure_scene_skeleton, write_json
from core.model_clients import ChatRequest
from person_attribute_search.bucket_queries import (
    build_chunk_query_buckets,
    build_query_context,
    generate_bucket_queries,
    normalize_buckets,
)


class _FakeClient:
    """Records the last request and returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_request: ChatRequest | None = None
        self.calls = 0

    def generate(self, request: ChatRequest) -> str:
        self.last_request = request
        self.calls += 1
        return self.response


class _SequenceClient:
    """Return canned responses in order, repeating the final response."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request: ChatRequest) -> str:
        del request
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_normalize_buckets_accepts_pairs_strings_and_dicts() -> None:
    parsed = {
        "PAS": [["person wearing red shirt", "top outer color"], "bare string", ""],
        "Anomaly": [{"query": "person falling", "source": "fall category"}],
        "Caption": [{"text": "warehouse aisle", "evidence": "scene"}],
        "Unknown": [["ignored", "x"]],
    }
    out = normalize_buckets(parsed)
    assert out["PAS"] == [
        ["person wearing red shirt", "top outer color"],
        ["bare string", ""],
    ]
    assert out["Anomaly"] == [["person falling", "fall category"]]
    assert out["Caption"] == [["warehouse aisle", "scene"]]
    assert "Unknown" not in out


def test_normalize_buckets_caps_per_bucket() -> None:
    parsed = {"PAS": [f"q{i}" for i in range(10)], "Anomaly": [], "Caption": []}
    out = normalize_buckets(parsed, per_bucket=3)
    assert len(out["PAS"]) == 3


def test_normalize_buckets_handles_non_dict() -> None:
    out = normalize_buckets(None)
    assert out == {"PAS": [], "Anomaly": [], "Caption": []}


def test_normalize_buckets_treats_non_sequence_bucket_as_empty() -> None:
    out = normalize_buckets({"PAS": "abc", "Anomaly": {"query": "x"}, "Caption": ["ok"]})
    assert out == {"PAS": [], "Anomaly": [], "Caption": [["ok", ""]]}


def test_build_query_context_includes_voted_categories() -> None:
    scene = {"parsed": {"scene_caption": "A warehouse aisle."}}
    dense = {"parsed": {"dense caption": "00:00 - 00:01: workers walking."}}
    anomaly_gt = {"voted_categories": ["person_falling_or_collapsing"]}
    context = build_query_context(scene, dense, anomaly_gt)
    assert "Scene context: A warehouse aisle." in context
    assert "person_falling_or_collapsing" in context
    assert "workers walking" in context


def test_build_query_context_normal_instruction_when_no_anomaly() -> None:
    anomaly_gt: dict[str, list[str]] = {"voted_categories": []}
    context = build_query_context("", "", anomaly_gt)
    assert "none; use normal visible behavior" in context


def test_generate_bucket_queries_parses_response_and_sends_context() -> None:
    response = json.dumps(
        {
            "PAS": [["person wearing yellow vest", "top outer"]],
            "Anomaly": [["person slipping on floor", "fall category"]],
            "Caption": [["busy warehouse aisle", "scene"]],
        }
    )
    client = _FakeClient(response)
    buckets = generate_bucket_queries(
        client,
        context_text="Anomaly categories present (multi-model majority): person_falling",
        people=[{"track_id": 0, "attributes": {"top outer color": "yellow"}}],
    )
    assert buckets["Anomaly"] == [["person slipping on floor", "fall category"]]
    assert buckets["Caption"] == [["busy warehouse aisle", "scene"]]
    assert client.last_request is not None
    assert "person_falling" in client.last_request.prompt
    assert "track_id" in client.last_request.prompt


def test_generate_bucket_queries_empty_on_unparseable_response() -> None:
    client = _FakeClient("not json at all")
    buckets = generate_bucket_queries(client, context_text="ctx", people=[])
    assert buckets == {"PAS": [], "Anomaly": [], "Caption": []}
    assert "No person attributes available." in (
        client.last_request.prompt if client.last_request else ""
    )


def test_generate_bucket_queries_retries_unusable_response() -> None:
    valid = json.dumps(
        {
            "PAS": [],
            "Anomaly": [["person falling", "fall category"]],
            "Caption": [["warehouse aisle", "scene caption"]],
        }
    )
    client = _SequenceClient(["not json", valid])

    buckets = generate_bucket_queries(
        client,
        context_text="ctx",
        people=[],
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    assert client.calls == 2
    assert buckets["Anomaly"] == [["person falling", "fall category"]]
    assert buckets["Caption"] == [["warehouse aisle", "scene caption"]]


def test_generate_bucket_queries_returns_last_response_after_retry_exhaustion() -> None:
    client = _SequenceClient(["not json"])

    buckets = generate_bucket_queries(
        client,
        context_text="ctx",
        people=[],
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    assert client.calls == 3
    assert buckets == {"PAS": [], "Anomaly": [], "Caption": []}


def test_generate_bucket_queries_interpolates_bucket_count() -> None:
    client = _FakeClient('{"PAS": [], "Anomaly": [], "Caption": []}')
    generate_bucket_queries(client, context_text="ctx", people=[], per_bucket=3)
    assert client.last_request is not None
    assert "Return exactly 3 queries for each bucket." in client.last_request.prompt


def test_build_chunk_query_buckets_skips_llm_when_context_artifacts_missing(
    tmp_path: Path,
) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    client = _FakeClient('{"PAS": [], "Anomaly": [], "Caption": []}')
    optional_failures: list[str] = []

    buckets = build_chunk_query_buckets(
        client,
        paths=paths,
        people=[],
        flat_queries=["person in vest"],
        video_captions_sidecars=("captioning/video_captions.json",),
        anomaly_items_sidecars=("visual_qa/items.json",),
        model_name="m",
        optional_failures=optional_failures,
    )

    assert buckets == {"PAS": [["person in vest", ""]], "Anomaly": [], "Caption": []}
    assert client.calls == 0
    assert optional_failures == [
        "bucket_query_context_missing: no captions/anomaly artifacts found"
    ]


def test_build_chunk_query_buckets_records_unusable_model_output(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    write_json(
        paths.sidecars_dir / "captioning" / "video_captions.json",
        {"summary": "A warehouse aisle.", "windows": []},
    )
    client = _FakeClient("not json")
    optional_failures: list[str] = []

    buckets = build_chunk_query_buckets(
        client,
        paths=paths,
        people=[],
        flat_queries=["person in vest"],
        video_captions_sidecars=("captioning/video_captions.json",),
        anomaly_items_sidecars=("visual_qa/items.json",),
        model_name="m",
        response_retries=1,
        response_retry_backoff_s=0.0,
        optional_failures=optional_failures,
    )

    assert buckets == {"PAS": [["person in vest", ""]], "Anomaly": [], "Caption": []}
    assert client.calls == 2
    assert optional_failures == [
        "bucket_query_generation_failed: LLM returned no usable Anomaly/Caption queries"
    ]
