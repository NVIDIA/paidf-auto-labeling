# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly classification adapter and sidecar builder.

This sub-package is the anomaly-mode counterpart to :mod:`reasoning.msted`:

- :mod:`reasoning.anomaly.llm` is the pure LLM adapter — it renders the
  prompt, calls the endpoint, and returns a normalized verdict dict. It
  never touches the filesystem and never validates against DAFT.
- :mod:`reasoning.anomaly.sidecar` turns that verdict into the on-disk
  ``sidecars/reasoning/anomaly.json`` payload, including the
  re-perception request a downstream captioning (VLM) pass can act on.

Anomaly output is deliberately a *sidecar*, not a DAFT contextual/task
type: the DAFT v3 type set is a closed enum and cannot host a new type
without a core-schema change. Keeping the verdict in a sidecar lets the
reasoning service emit it today while still feeding the causal-linkage
stage's ``video_type`` auto-tagging.
"""

from __future__ import annotations

from reasoning.anomaly.llm import AnomalyLLMError, classify_anomaly_with_llm
from reasoning.anomaly.sidecar import ANOMALY_SIDECAR_SCHEMA, to_anomaly_sidecar

__all__ = [
    "ANOMALY_SIDECAR_SCHEMA",
    "AnomalyLLMError",
    "classify_anomaly_with_llm",
    "to_anomaly_sidecar",
]
