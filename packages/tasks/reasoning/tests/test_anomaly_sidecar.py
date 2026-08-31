# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the anomaly sidecar payload builder (pure shaping logic)."""

from __future__ import annotations

from reasoning.anomaly.sidecar import ANOMALY_SIDECAR_SCHEMA, to_anomaly_sidecar


class TestNormalScene:
    def test_minimal_normal_payload(self):
        verdict = {"classification": "normal", "confidence": "high", "reasoning": "routine"}
        out = to_anomaly_sidecar(verdict)
        assert out["schema"] == ANOMALY_SIDECAR_SCHEMA
        assert out["classification"] == "normal"
        assert out["confidence"] == "high"
        assert out["reasoning"] == "routine"
        # Normal scenes carry no anomaly-only signals or hand-off request.
        assert "root_cause" not in out
        assert "highlight" not in out
        assert "reperception_request" not in out

    def test_normal_scene_never_requests_reperception(self):
        # Even if a malformed verdict smuggles a highlight onto a normal
        # scene, the builder must not emit a re-perception request.
        verdict = {
            "classification": "normal",
            "reasoning": "x",
            "highlight": {"start": "00:00", "end": "00:02"},
        }
        out = to_anomaly_sidecar(verdict)
        assert "reperception_request" not in out
        assert "highlight" not in out


class TestAnomalyScene:
    def test_anomaly_with_highlight_emits_reperception_request(self):
        verdict = {
            "classification": "anomaly",
            "confidence": "medium",
            "reasoning": "collision",
            "root_cause": "loss of control",
            "consequence": "barrier impact",
            "highlight": {"start": "00:04", "end": "00:07", "description": "impact"},
        }
        out = to_anomaly_sidecar(verdict, model="my-model", sources=["sidecars/metadata.json"])
        assert out["classification"] == "anomaly"
        assert out["root_cause"] == "loss of control"
        assert out["consequence"] == "barrier impact"
        assert out["highlight"] == {
            "start": "00:04",
            "end": "00:07",
            "description": "impact",
        }
        request = out["reperception_request"]
        assert request["status"] == "requested"
        assert request["target"] == "captioning"
        assert request["reason"] == "anomaly_highlight"
        assert request["window"] == {"start": "00:04", "end": "00:07"}
        assert request["description"] == "impact"
        assert out["model"] == "my-model"
        assert out["sources"] == ["sidecars/metadata.json"]

    def test_highlight_extra_keys_do_not_leak(self):
        # A richer/cached verdict may smuggle extra keys onto the
        # highlight; only start/end/description are persisted.
        verdict = {
            "classification": "anomaly",
            "reasoning": "collision",
            "highlight": {
                "start": "00:04",
                "end": "00:07",
                "description": "impact",
                "internal_score": 0.97,
                "model_notes": "leak me",
            },
        }
        out = to_anomaly_sidecar(verdict)
        assert out["highlight"] == {
            "start": "00:04",
            "end": "00:07",
            "description": "impact",
        }
        assert "internal_score" not in out["highlight"]
        assert "model_notes" not in out["highlight"]
        assert set(out["reperception_request"]["window"]) == {"start", "end"}

    def test_anomaly_without_highlight_has_no_request(self):
        verdict = {
            "classification": "anomaly",
            "reasoning": "something off but not localized",
            "root_cause": "unclear",
        }
        out = to_anomaly_sidecar(verdict)
        assert out["classification"] == "anomaly"
        assert out["root_cause"] == "unclear"
        assert "highlight" not in out
        assert "reperception_request" not in out
