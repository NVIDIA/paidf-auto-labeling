# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pipeline as pipeline_mod
from nvcf_msc_utils import is_remote_path

_log_plan = pipeline_mod._log_plan
_append_pipeline_log = pipeline_mod._append_pipeline_log
_empty_policy_value = pipeline_mod._empty_policy_value


def test_empty_policy_value():
    assert _empty_policy_value("warn", "warn") == "warn"
    assert _empty_policy_value("fail", "warn") == "fail"
    assert _empty_policy_value("WARN", "fail") == "warn"
    assert _empty_policy_value("", "warn") == "warn"
    assert _empty_policy_value(None, "fail") == "fail"
    assert _empty_policy_value("unknown", "warn") == "warn"


def test_is_remote_path():
    assert is_remote_path("s3://bucket/file")
    assert is_remote_path("http://site.com/file")
    assert is_remote_path("https://site.com/file")
    assert is_remote_path("gs://bucket/file")
    assert is_remote_path("msc://bucket/file")
    assert not is_remote_path("/local/file")
    assert not is_remote_path("./file")
