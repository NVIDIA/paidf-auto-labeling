# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from mcq_generation.mcq.utils.aggregation import aggregate_window_mcqs
from mcq_generation.mcq.utils.bank import (
    collect_embedded_bank_from_prompt,
    filter_mcq_items_strict,
    options_map_from_bank,
    read_bank,
)
from mcq_generation.mcq.utils.ids import present_ids


def test_read_bank_dedupes_exact_options_before_prompting(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(
        json.dumps(
            {
                "name": "dupes",
                "questions": [
                    {
                        "id": "q1",
                        "question": "?",
                        "options": ["Yes", "No", "Yes"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bank = read_bank(path)

    assert bank["questions"][0]["options"] == ["Yes", "No"]


def test_read_bank_dedupes_prefixed_options_by_display_value(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(
        json.dumps(
            {
                "name": "dupes",
                "questions": [
                    {
                        "id": "q1",
                        "question": "?",
                        "options": ["A. b", "B. b", "C. c"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bank = read_bank(path)

    assert bank["questions"][0]["options"] == ["A. b", "B. c"]


def test_embedded_bank_and_options_map_use_normalized_options():
    prompt = """
```json
{"questions": [{"id": "q1", "question": "?", "options": ["A.b", "B.b", "C.c"]}]}
```
"""

    bank = collect_embedded_bank_from_prompt(prompt)

    assert bank is not None
    assert bank["questions"][0]["options"] == ["A. b", "B. c"]
    assert options_map_from_bank(bank) == {"q1": ["A. b", "B. c"]}


def test_embedded_bank_accepts_missing_options_for_open_qa():
    prompt = """
```json
{"questions": [{"id": "q1", "question": "Describe what happened."}]}
```
"""

    bank = collect_embedded_bank_from_prompt(prompt)

    assert bank is not None
    assert bank["questions"][0] == {"id": "q1", "question": "Describe what happened."}
    assert options_map_from_bank(bank) == {"q1": []}


def test_filter_accepts_current_option_aliases():
    bank = {
        "questions": [
            {
                "id": "q1",
                "question": "?",
                "options": ["A. b", "B. c"],
            }
        ]
    }

    filtered = filter_mcq_items_strict(
        [
            {"id": "q1", "question": "?", "answer": "A"},
            {"id": "q1", "question": "?", "answer": "b"},
            {"id": "q1", "question": "?", "answer": "B. c"},
        ],
        include_if_map={},
        options_map=options_map_from_bank(bank),
    )

    assert filtered == [
        {"id": "q1", "question": "?", "answer": "A"},
        {"id": "q1", "question": "?", "answer": "b"},
        {"id": "q1", "question": "?", "answer": "B. c"},
    ]


def test_filter_rejects_removed_prefixed_duplicate_alias():
    bank = {
        "questions": [
            {
                "id": "q1",
                "question": "?",
                "options": ["A. b", "B. c"],
            }
        ]
    }

    filtered = filter_mcq_items_strict(
        [{"id": "q1", "question": "?", "answer": "B. b"}],
        include_if_map={},
        options_map=options_map_from_bank(bank),
    )

    assert filtered == []


def test_filter_rejects_missing_answer_for_open_qa():
    bank = {
        "questions": [
            {
                "id": "q1",
                "question": "Describe the scene.",
                "options": [],
            }
        ]
    }

    filtered = filter_mcq_items_strict(
        [
            {"id": "q1", "question": "Describe the scene.", "options": []},
            {"id": "q1", "question": "Describe the scene.", "options": [], "answer": ""},
        ],
        include_if_map={},
        options_map=options_map_from_bank(bank),
    )

    assert filtered == []


def test_present_ids_requires_non_empty_answer():
    obj = {
        "mcq": [
            {"id": "q1", "question": "Describe the scene.", "options": []},
            {"id": "q2", "question": "Describe the road.", "options": [], "answer": "A divided road."},
        ]
    }

    assert present_ids(obj) == ["q2"]


def test_aggregation_rejects_removed_prefixed_duplicate_alias():
    out = aggregate_window_mcqs(
        [
            {
                "mcq": [
                    {
                        "id": "q1",
                        "question": "?",
                        "options": ["A. b", "B. c"],
                        "answer": "B. b",
                    }
                ]
            },
            {
                "mcq": [
                    {
                        "id": "q1",
                        "question": "?",
                        "options": ["A. b", "B. c"],
                        "answer": "A. b",
                    }
                ]
            },
        ],
        video_id="main",
    )

    assert out["mcq"] == []
