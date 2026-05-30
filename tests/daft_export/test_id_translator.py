# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest
from daft_export.id_translator import (
    ground_vlm_ids,
    index_instances_by_suffix,
    strip_ungrounded_id_annotations,
)


@pytest.fixture
def quiet_logger() -> logging.Logger:
    # Tests inspect log records via caplog; the logger itself just needs to exist.
    return logging.getLogger("test.id_translator")


# Realistic tracker keys mimicking what `rfdetr_tracking.py` writes
# (`f"{class_name}_{track_id}"`). Mixes single and multi-word class prefixes
# so we cover the rsplit-on-last-underscore parsing path.
REAL_TRACKER_KEYS = [
    "car_1",
    "car_15",
    "car_76",
    "car_103",
    "motorcycle_159",
    "traffic_light_3",
    "person_42",
]


class TestIndexInstancesBySuffix:
    def test_indexes_by_trailing_int(self):
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        assert idx["1"] == "car_1"
        assert idx["76"] == "car_76"
        assert idx["159"] == "motorcycle_159"
        assert idx["3"] == "traffic_light_3"

    def test_collision_last_wins(self):
        # Mid-track class flip would put two keys at the same suffix; we
        # accept dict-overwrite semantics. Ambiguity is rare and the right
        # answer is undefined either way (VLM only saw the number).
        idx = index_instances_by_suffix(["car_76", "truck_76"])
        assert idx["76"] == "truck_76"

    def test_drops_non_integer_suffixes(self):
        idx = index_instances_by_suffix(["car_x", "car_7", "no_underscore", "trailing_"])
        assert idx == {"7": "car_7"}

    def test_drops_non_strings(self):
        # Defensive: instances.json comes from JSON, but coerce just in case.
        idx = index_instances_by_suffix(["car_7", 42, None])  # type: ignore[list-item]
        assert idx == {"7": "car_7"}

    def test_empty_input(self):
        assert index_instances_by_suffix([]) == {}


class TestGroundVlmIds:
    def test_translates_unique_matches(self, quiet_logger):
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        out = ground_vlm_ids(["id_1", "id_76", "id_159"], idx, logger=quiet_logger)
        assert out == ["car_1", "car_76", "motorcycle_159"]

    def test_preserves_order(self, quiet_logger):
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        out = ground_vlm_ids(["id_103", "id_1", "id_76"], idx, logger=quiet_logger)
        assert out == ["car_103", "car_1", "car_76"]

    def test_normalizes_leading_zero_suffixes(self, quiet_logger):
        idx = index_instances_by_suffix(["person_1", "person_3", "person_5"])
        out = ground_vlm_ids(["id_001", "id_0003", "id_5"], idx, logger=quiet_logger)
        assert out == ["person_1", "person_3", "person_5"]

    def test_long_numeric_suffix_does_not_crash(self, quiet_logger):
        idx = index_instances_by_suffix(["person_1"])
        out = ground_vlm_ids(["id_" + ("0" * 5000) + "1"], idx, logger=quiet_logger)
        assert out == ["person_1"]

    def test_accepts_bare_digit_values(self, quiet_logger):
        idx = index_instances_by_suffix(["person_1", "person_3", "person_5"])
        out = ground_vlm_ids(["1", "03", 5], idx, logger=quiet_logger)
        assert out == ["person_1", "person_3", "person_5"]

    def test_drops_unground_with_warning(self, quiet_logger, caplog):
        # Mismatch case: id_78 has no <class>_78 in the catalogue. Real
        # matches alongside it must survive (translator never drops grounded ids).
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        with caplog.at_level(logging.WARNING, logger=quiet_logger.name):
            out = ground_vlm_ids(["id_1", "id_78", "id_103"], idx, logger=quiet_logger)
        assert out == ["car_1", "car_103"]
        warnings = [r for r in caplog.records if "ungrounded" in r.message]
        assert len(warnings) == 1
        assert "id_78" in warnings[0].args  # type: ignore[operator]

    def test_empty_catalogue_drops_silently(self, quiet_logger, caplog):
        # When the catalogue is empty we skip per-id warnings — the caller
        # logs once at the call site to avoid spam.
        with caplog.at_level(logging.WARNING, logger=quiet_logger.name):
            out = ground_vlm_ids(["id_6131", "id_6127"], {}, logger=quiet_logger)
        assert out == []
        assert not [r for r in caplog.records if "ungrounded" in r.message]

    @pytest.mark.parametrize(
        "bad",
        [
            "car_76",  # missing the id_ prefix (a tracker-shaped string slipped in)
            "id_",  # prefix only
            "id_abc",  # non-integer suffix
            "id_7.5",  # not an integer
            -1,  # negative integer
            True,  # bool is not a track id even though it is int-like
            "",  # empty
        ],
    )
    def test_skips_malformed(self, quiet_logger, bad):
        # Defensive: VLM contract is id_<int>, but if upstream slips up we
        # silently skip rather than crash. Per-id warning is reserved for
        # ungrounded suffixes; malformed strings are noise from a different
        # problem class entirely.
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        out = ground_vlm_ids([bad, "id_1"], idx, logger=quiet_logger)
        assert out == ["car_1"]

    def test_empty_input(self, quiet_logger):
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        assert ground_vlm_ids([], idx, logger=quiet_logger) == []

    def test_logger_optional(self):
        # The function must work without a logger (callers in tests / scripts
        # that don't care about diagnostics).
        idx = index_instances_by_suffix(REAL_TRACKER_KEYS)
        assert ground_vlm_ids(["id_1", "id_999"], idx) == ["car_1"]


class TestStripUngroundedIdAnnotations:
    """Prose-side companion to ``ground_vlm_ids``. Same grounding rule
    applied to ``{id: <n>}`` annotations the VLM leaks into prose fields
    (``event_caption`` / ``scene_description`` / ``event_summary``) by
    copying numbers out of the prompt's few-shot examples."""

    def test_drops_ungrounded_annotation(self):
        idx = index_instances_by_suffix(["car_7"])
        out = strip_ungrounded_id_annotations(
            "white sedan {id: 6131} stopped at the intersection",
            idx,
        )
        assert out == "white sedan stopped at the intersection"

    def test_keeps_grounded_annotation(self):
        idx = index_instances_by_suffix(["car_7"])
        out = strip_ungrounded_id_annotations(
            "white sedan {id: 7} stopped at the intersection",
            idx,
        )
        assert out == "white sedan {id: 7} stopped at the intersection"

    def test_keeps_grounded_annotation_with_leading_zero(self):
        idx = index_instances_by_suffix(["car_7"])
        out = strip_ungrounded_id_annotations(
            "white sedan {id: 007} stopped at the intersection",
            idx,
        )
        assert out == "white sedan {id: 007} stopped at the intersection"

    def test_empty_catalogue_drops_all(self):
        # det/track OFF: no overlay was rendered, so any ``{id: <n>}`` the
        # VLM emits is necessarily a leak from the few-shot examples.
        out = strip_ungrounded_id_annotations(
            "motorcycle {id: 6127} and sedan {id: 6131} collided",
            {},
        )
        assert out == "motorcycle and sedan collided"

    def test_keeps_mixed_annotation_with_any_grounded(self):
        # ``{id: 6131, id: 7}`` — partial rewrite would distort the model's
        # intent, so we keep the whole annotation if any inner id is in the
        # catalogue. Documented contract; matches the docstring.
        idx = index_instances_by_suffix(["car_7"])
        out = strip_ungrounded_id_annotations(
            "vehicles {id: 6131, id: 7} moving through",
            idx,
        )
        assert out == "vehicles {id: 6131, id: 7} moving through"

    def test_drops_mixed_annotation_with_no_grounded(self):
        idx = index_instances_by_suffix(["car_99"])
        out = strip_ungrounded_id_annotations(
            "vehicles {id: 6131, id: 6145} moving through",
            idx,
        )
        assert out == "vehicles moving through"

    def test_no_annotations_passthrough(self):
        idx = index_instances_by_suffix(["car_7"])
        out = strip_ungrounded_id_annotations(
            "Plain prose with no annotations.",
            idx,
        )
        assert out == "Plain prose with no annotations."

    def test_non_string_passthrough(self):
        # Defensive: callers shouldn't pass non-strings, but if they do we
        # return as-is rather than crash.
        assert strip_ungrounded_id_annotations(None, {}) is None  # type: ignore[arg-type]
        assert strip_ungrounded_id_annotations(42, {}) == 42  # type: ignore[arg-type]

    def test_no_double_spaces_after_strip_mid_sentence(self):
        # Leading whitespace before the annotation is part of the regex
        # match so removal doesn't leave ``"sedan  stopped"`` artifacts.
        out = strip_ungrounded_id_annotations(
            "sedan {id: 6131} stopped",
            {},
        )
        assert out == "sedan stopped"
        assert "  " not in out

    def test_no_leading_whitespace_after_strip_at_start(self):
        # When the annotation starts the string, the regex match has no
        # leading whitespace to consume; the trailing ``.lstrip()`` cleans
        # up the orphaned space that would otherwise survive.
        out = strip_ungrounded_id_annotations(
            "{id: 6131} starts the sentence",
            {},
        )
        assert out == "starts the sentence"

    def test_realistic_prompt_leak_example1(self):
        # Verbatim shape we observed leaking in the cp_output_video OFF
        # run (regression bait for the fix). Numbers are from the prompt's
        # Example 1 few-shot block.
        out = strip_ungrounded_id_annotations(
            "POST-ACCIDENT SCENE: motorcycle {id: 6127} on its side, "
            "rider {id: 6245} on the ground, sedan {id: 6131} stopped.",
            {},
        )
        assert "{id:" not in out
        assert "  " not in out
        assert "motorcycle" in out and "sedan" in out and "rider" in out

    def test_realistic_prompt_leak_example5_multi_id_annotation(self):
        # The literal multi-id annotation shape from prompt Example 5:
        # ``{id: 6131, id: 6145, id: 6190, id: 6202}``. All ungrounded → the
        # entire annotation goes; surrounding prose stays.
        out = strip_ungrounded_id_annotations(
            "Four vehicles {id: 6131, id: 6145, id: 6190, id: 6202} moving through intersection",
            {},
        )
        assert out == "Four vehicles moving through intersection"

    def test_idempotent(self):
        # Running the cleanup twice on the same text yields the same
        # result as running it once — important if the same prose flows
        # through multiple converters.
        idx = index_instances_by_suffix(["car_7"])
        text = "sedan {id: 6131} and pickup {id: 7} together"
        once = strip_ungrounded_id_annotations(text, idx)
        twice = strip_ungrounded_id_annotations(once, idx)
        assert once == twice
        assert once == "sedan and pickup {id: 7} together"
