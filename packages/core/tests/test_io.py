# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from core import read_json, read_jsonl, write_json, write_jsonl


def test_write_and_read_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "file.json"
    payload = {"a": 1, "b": [2, 3]}
    write_json(p, payload)
    assert p.read_text(encoding="utf-8").endswith("\n")
    assert read_json(p) == payload


def test_write_and_read_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "lines.jsonl"
    records = [{"i": 0}, {"i": 1}, {"i": 2}]
    write_jsonl(p, records)
    assert list(read_jsonl(p)) == records
