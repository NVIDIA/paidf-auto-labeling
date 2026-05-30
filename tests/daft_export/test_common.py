# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re

import pytest
from daft_export.common import DAFT_VERSION, DaftConvertError, metadata_block, write_daft_json


class TestDaftConvertError:
    def test_subclasses_value_error(self):
        # Downstream can catch ValueError as a broader net.
        assert issubclass(DaftConvertError, ValueError)


class TestMetadataBlock:
    def test_minimum(self):
        block = metadata_block("mcq")
        assert block["type"] == "mcq"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", block["date"])
        assert set(block.keys()) == {"type", "date"}

    def test_explicit_date(self):
        block = metadata_block("bcq", iso_date="2026-04-20")
        assert block["date"] == "2026-04-20"

    def test_with_optional_fields(self):
        block = metadata_block(
            "mcq",
            iso_date="2026-04-20",
            description="trial run",
            license_str="MIT",
            tags=["traffic", "accident"],
        )
        assert block == {
            "type": "mcq",
            "date": "2026-04-20",
            "description": "trial run",
            "license": "MIT",
            "tags": ["traffic", "accident"],
        }

    def test_omits_unset_optionals(self):
        # DAFT uses `additionalProperties: false`, so absent keys must stay absent.
        block = metadata_block("mcq", iso_date="2026-04-20")
        assert "description" not in block
        assert "license" not in block
        assert "tags" not in block


class TestWriteDaftJson:
    def test_writes_payload(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"version": DAFT_VERSION, "metadata": {"type": "mcq"}, "items": [{"x": 1}]}
        write_daft_json(path, payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "out.json"
        write_daft_json(path, {"version": DAFT_VERSION, "metadata": {"type": "mcq"}})
        assert path.exists()

    def test_rejects_non_dict(self, tmp_path):
        with pytest.raises(TypeError, match="expects a dict"):
            write_daft_json(tmp_path / "x.json", ["not", "a", "dict"])

    def test_rejects_missing_version(self, tmp_path):
        with pytest.raises(ValueError, match="missing 'version'"):
            write_daft_json(tmp_path / "x.json", {"metadata": {"type": "mcq"}})

    def test_rejects_missing_metadata(self, tmp_path):
        with pytest.raises(ValueError, match="missing 'metadata'"):
            write_daft_json(tmp_path / "x.json", {"version": DAFT_VERSION})
