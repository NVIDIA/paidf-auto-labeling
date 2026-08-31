# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PAS-only service wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from core import DataEntry, ensure_scene_skeleton, write_json
from event_and_person_attribute_search_service.main import (
    EventAndPersonAttributeSearchService,
    _apply_cli_overrides,
    _load_config,
    _MeasuredTask,
    main,
)
from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.task import PersonAttributeSearchTask


def test_load_config_accepts_composite_section(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "event_and_person_attribute_search:\n"
        "  captioning:\n"
        "    enabled: true\n"
        "  person_attribute_search:\n"
        "    dataset: test\n"
        "    easy_count: 3\n",
        encoding="utf-8",
    )

    config = _load_config(path)

    assert config.dataset == "test"
    assert config.easy_count == 3


def test_load_config_accepts_direct_pas_section(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"person_attribute_search":{"dataset":"direct"}}', encoding="utf-8")

    assert _load_config(path).dataset == "direct"


def test_image_attribute_cookbook_loads_bundle_pas_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = (
        repo_root
        / "cookbooks"
        / "visual_attribute_search"
        / "configs"
        / "pipeline_image_attributes_pas.yaml"
    )

    config = _load_config(path)

    assert config.dataset == "upa"
    assert config.bundle_query_generation is True
    assert config.bundle_query_count == 3
    assert config.llm_query_generation is False
    assert config.bucket_query_generation is False
    assert config.emit_contextual is False
    assert config.emit_daft_contextual is False
    assert config.attribute_json is None
    assert config.query_prompt_file is None


def test_bundle_max_tokens_default_leaves_headroom_for_16k_context_models() -> None:
    """Default must not consume an entire 16k context window (bug 6508754)."""
    config = PersonAttributeSearchConfig()
    assert config.bundle_max_tokens == 2048
    assert config.bundle_max_tokens < 16384


def test_image_attribute_cookbook_passes_safe_max_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = (
        repo_root
        / "cookbooks"
        / "visual_attribute_search"
        / "configs"
        / "pipeline_image_attributes_pas.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    args = payload["workflow"]["nodes"]["person_attribute_search"]["args"]
    assert "--max-tokens" in args
    max_tokens = int(args[args.index("--max-tokens") + 1])
    assert max_tokens == 2048
    assert max_tokens < 16384


def test_load_config_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        _load_config(path)


def test_cli_surface_has_no_captioning_or_visual_qa_options() -> None:
    parser = argparse.ArgumentParser()
    EventAndPersonAttributeSearchService().add_service_args(parser)
    options = parser.format_help()

    assert "--attribute-json" in options
    assert "--query-prompt-file" in options
    assert "--llm-retries" in options
    assert "--llm-response-retries" in options
    assert "--vlm-endpoint-url" not in options
    assert "--person-question-bank-file" not in options
    assert "--overwrite" not in options


def test_cli_prompt_enables_bundle_generation() -> None:
    args = argparse.Namespace(
        attribute_json="attributes.json",
        query_prompt_file="prompt.json",
        query_prompt_text=None,
        query_count=3,
        llm_provider="openai-compatible",
        llm_endpoint_url="http://model/v1",
        llm_model="model",
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        llm_retries=4,
        llm_retry_backoff_s=3.0,
        llm_response_retries=5,
        llm_response_retry_backoff_s=1.5,
        use_template_for_medium=False,
    )

    config = _apply_cli_overrides(PersonAttributeSearchConfig(), args)

    assert config.attribute_json == "attributes.json"
    assert config.bundle_query_generation is True
    assert config.bundle_query_count == 3
    assert config.llm_temperature == 0.2
    assert config.llm_retries == 4
    assert config.llm_retry_backoff_s == 3.0
    assert config.llm_response_retries == 5
    assert config.llm_response_retry_backoff_s == 1.5
    assert config.bundle_max_tokens == 2048


def test_measured_task_wraps_only_pas_task(tmp_path: Path) -> None:
    scene = ensure_scene_skeleton(tmp_path / "scene")
    write_json(
        scene.sidecars_dir / "visual_qa" / "items.json",
        {"items": [{"id": "top outer color", "answer": "red"}]},
    )
    entry = DataEntry(media_path="person.jpg", data_path=str(scene.scene_dir))
    task = _MeasuredTask(PersonAttributeSearchTask(PersonAttributeSearchConfig(write_hitl=False)))

    assert task.run(entry) == entry
    assert task.delegate.name == "person_attribute_search"
    assert task.entry_reports[entry.id].success is True


def test_attribute_json_requires_one_service_entry(tmp_path: Path) -> None:
    attribute_json = tmp_path / "attributes.json"
    attribute_json.write_text('{"attributes":{"top_outer_color":"red"}}', encoding="utf-8")
    service = EventAndPersonAttributeSearchService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args(["--attribute-json", str(attribute_json)])
    entries = [
        DataEntry(media_path="a.jpg", data_path=str(tmp_path / "a")),
        DataEntry(media_path="b.jpg", data_path=str(tmp_path / "b")),
    ]

    with pytest.raises(ValueError, match="exactly one service input entry"):
        service.execute(args, entries)


def test_main_runs_service() -> None:
    with patch.object(EventAndPersonAttributeSearchService, "run") as run:
        main()
    run.assert_called_once_with()
