# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from reasoning.prompts import (
    BUNDLED_PROMPT_DIR,
    PromptError,
    PromptRegistry,
    PromptVariant,
)


def write_yaml(path: Path, **fields) -> Path:
    """Write a minimal prompt YAML file. Caller controls every field so
    tests stay explicit about what each variant declares."""
    lines = []
    for key, val in fields.items():
        if "\n" in str(val):
            indented = textwrap.indent(str(val), "  ")
            lines.append(f"{key}: |\n{indented}")
        else:
            lines.append(f"{key}: {val!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestBundled:
    def test_bundled_dir_exists(self):
        # Sanity check: the bundled directory ships with the package.
        assert BUNDLED_PROMPT_DIR.is_dir(), (
            f"bundled prompt dir {BUNDLED_PROMPT_DIR} missing — should be created when the "
            "first prompt YAML is added"
        )

    def test_bundled_includes_msted_default(self):
        reg = PromptRegistry()
        assert "msted_default" in reg.names()

    def test_bundled_msted_default_renders(self):
        reg = PromptRegistry()
        v = reg.get("msted_default")
        # Required placeholders the LLM caller fills in.
        rendered = v.render_user(
            scene_description_block="Scene description:\nFoo.",
            windows_block="00:00-00:04: caption A\n00:04-00:07: caption B",
            event_summary_block="",
        )
        assert "caption A" in rendered
        assert "caption B" in rendered


class TestExtraDirsOverride:
    def test_extra_dir_overrides_bundled_variant(self, tmp_path: Path):
        # Caller-supplied YAML with the same name overrides the bundled one.
        # This is the documented use-case-agnostic extension hook: ship a
        # domain-specific variant by putting <name>.yaml in your prompt_dir.
        write_yaml(
            tmp_path / "msted_default.yaml",
            system="OVERRIDE system",
            user_template="OVERRIDE template windows={windows_block}",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path])
        v = reg.get("msted_default")
        # Block scalars round-trip with a trailing newline; the substring
        # check is the contract, the exact whitespace isn't.
        assert "OVERRIDE system" in v.system
        rendered = v.render_user(windows_block="abc")
        assert "OVERRIDE template windows=abc" in rendered

    def test_extra_dir_can_add_new_variant(self, tmp_path: Path):
        write_yaml(
            tmp_path / "av_surveillance.yaml",
            system="AV system",
            user_template="AV {windows_block}",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path])
        assert "av_surveillance" in reg.names()
        # bundled defaults still present
        assert "msted_default" in reg.names()

    def test_multiple_extra_dirs_layered_in_order(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        write_yaml(
            first / "shared.yaml",
            system="from first",
            user_template="A",
        )
        write_yaml(
            second / "shared.yaml",
            system="from second",
            user_template="B",
        )
        reg = PromptRegistry(extra_dirs=[first, second])
        # second wins on collision
        assert "from second" in reg.get("shared").system
        assert "from first" not in reg.get("shared").system

    def test_missing_extra_dir_raises(self, tmp_path: Path):
        with pytest.raises(PromptError, match="does not exist"):
            PromptRegistry(extra_dirs=[tmp_path / "nope"])


class TestNoBundled:
    def test_include_bundled_false_excludes_default(self, tmp_path: Path):
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert "msted_default" not in reg.names()

    def test_include_bundled_false_with_caller_only(self, tmp_path: Path):
        write_yaml(
            tmp_path / "only.yaml",
            system="S",
            user_template="U",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert reg.names() == ["only"]


class TestYamlValidation:
    def test_yaml_must_be_mapping(self, tmp_path: Path):
        (tmp_path / "bad.yaml").write_text("- a list at the top level\n", encoding="utf-8")
        with pytest.raises(PromptError, match="mapping at the top level"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_unknown_key_rejected(self, tmp_path: Path):
        write_yaml(
            tmp_path / "weird.yaml",
            system="S",
            user_template="U",
            mystery_field="boo",
        )
        with pytest.raises(PromptError, match="unknown keys"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_missing_system_rejected(self, tmp_path: Path):
        write_yaml(tmp_path / "no_sys.yaml", user_template="U")
        with pytest.raises(PromptError, match="system"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_missing_user_template_rejected(self, tmp_path: Path):
        write_yaml(tmp_path / "no_user.yaml", system="S")
        with pytest.raises(PromptError, match="user_template"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_empty_system_rejected(self, tmp_path: Path):
        write_yaml(tmp_path / "empty.yaml", system="   ", user_template="U")
        with pytest.raises(PromptError, match="system"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_invalid_yaml_rejected(self, tmp_path: Path):
        (tmp_path / "bad.yaml").write_text("system: 'unclosed\n", encoding="utf-8")
        with pytest.raises(PromptError, match="not valid YAML"):
            PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)

    def test_underscore_files_skipped(self, tmp_path: Path):
        # Files starting with _ are reserved for fragments / helpers.
        (tmp_path / "_helper.yaml").write_text(
            "this is intentionally invalid: : :", encoding="utf-8"
        )
        write_yaml(
            tmp_path / "real.yaml",
            system="S",
            user_template="U",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert reg.names() == ["real"]

    def test_non_yaml_files_skipped(self, tmp_path: Path):
        (tmp_path / "readme.txt").write_text("not a yaml file", encoding="utf-8")
        write_yaml(
            tmp_path / "real.yaml",
            system="S",
            user_template="U",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert reg.names() == ["real"]

    def test_yml_extension_accepted(self, tmp_path: Path):
        write_yaml(
            tmp_path / "alt.yml",
            system="S",
            user_template="U",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert reg.names() == ["alt"]


class TestLookup:
    def test_get_unknown_raises_with_known_names(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="known variants"):
            reg.get("does_not_exist")

    def test_source_returns_path(self, tmp_path: Path):
        path = write_yaml(
            tmp_path / "v.yaml",
            system="S",
            user_template="U",
        )
        reg = PromptRegistry(extra_dirs=[tmp_path], include_bundled=False)
        assert reg.source("v") == path

    def test_source_unknown_raises(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError):
            reg.source("nope")


class TestNewBundledPromptsShipped:
    """Sanity checks that the four new opt-in stage prompts ship with the
    package and contain the contract sentinels (strict-JSON instruction,
    answer-shape hint, etc.). These prompts are use-case agnostic by
    design — the sentinels here are *shape* guarantees, not domain
    vocabulary.

    Each prompt is also rendered with its real placeholders to catch
    template typos before they hit the LLM."""

    def test_open_qa_default_present(self):
        reg = PromptRegistry()
        v = reg.get("open_qa_default")
        assert "strict JSON" in v.system or "JSON only" in v.system
        rendered = v.render_user(
            windows_block="00:00-00:04: cap A",
            scene_description_block="Scene description:\nFoo.",
            event_summary_block="",
            questions_block="1. What happens?",
        )
        assert "What happens?" in rendered
        assert "items" in rendered

    def test_mcq_openended_default_present(self):
        reg = PromptRegistry()
        v = reg.get("mcq_openended_default")
        # Must instruct the LLM about the leading-letter answer shape
        # the converter enforces.
        assert "letter" in v.user_template.lower()
        assert "INCORRECT" in v.user_template  # CORRECT/INCORRECT example block
        rendered = v.render_user(
            windows_block="00:00-00:04: cap A",
            scene_description_block="",
            event_summary_block="",
            questions_block="1. Pick A or B\n   A: car\n   B: truck",
        )
        assert "Pick A or B" in rendered

    def test_bcq_openended_default_present(self):
        reg = PromptRegistry()
        v = reg.get("bcq_openended_default")
        # Must instruct the LLM about the Yes/No leading verdict the
        # converter enforces.
        assert "Yes" in v.user_template and "No" in v.user_template
        assert "INCORRECT" in v.user_template
        rendered = v.render_user(
            windows_block="00:00-00:04: cap A",
            scene_description_block="",
            event_summary_block="",
            questions_block="1. Did X happen?",
        )
        assert "Did X happen?" in rendered

    def test_causal_linkage_default_present(self):
        reg = PromptRegistry()
        v = reg.get("causal_linkage_default")
        # Must instruct the LLM about timecode shape (single MM:SS
        # values, never concatenated) — this is the same lesson MSTED
        # learned the hard way.
        assert "single timecodes" in v.user_template or "single value" in v.user_template
        assert "video_type" in v.user_template
        rendered = v.render_user(
            windows_block="00:00-00:04: cap A",
            scene_description_block="",
            pairs_block="1. t1=00:01, t2=00:05\n   question: Why?",
        )
        assert "00:01" in rendered and "00:05" in rendered

    def test_anomaly_classify_default_present(self):
        reg = PromptRegistry()
        v = reg.get("anomaly_classify_default")
        # Strict-JSON contract + the closed anomaly/normal verdict the
        # adapter normalizes against. These are shape guarantees, not
        # domain vocabulary — the prompt stays use-case neutral.
        assert "strict JSON" in v.system or "JSON only" in v.system
        assert "anomaly" in v.user_template and "normal" in v.user_template
        assert "classification" in v.user_template
        # Highlight window doubles as the re-perception request; single
        # timecodes only, never concatenated (the MSTED lesson again).
        assert "single timecode" in v.user_template
        # The template also carries an optional person-attribute evidence
        # block fed from person-attribute search (kept use-case neutral).
        assert "{person_attributes_block}" in v.user_template
        rendered = v.render_user(
            windows_block="00:00-00:04: cap A",
            scene_description_block="Scene description:\nFoo.",
            person_attributes_block=(
                "Person attributes (from person-attribute search):\n"
                "Person track 0: potential anomaly=no"
            ),
        )
        assert "cap A" in rendered
        assert "classification" in rendered
        assert "Person track 0" in rendered

    def test_reasoning_prompts_ship_length_examples(self):
        # The reference auto-label pipeline anchors CoT length with a
        # worked example in every reasoning prompt; the model imitates the
        # example length. These four prompts must each carry one so the
        # length caps are demonstrated, not just described.
        reg = PromptRegistry()
        for name in (
            "anomaly_classify_default",
            "causal_linkage_default",
            "open_qa_default",
            "temporal_localization_default",
        ):
            template = reg.get(name).user_template
            assert "Example of a CORRECT" in template, (
                f"{name} is missing its worked length example"
            )


class TestRender:
    def test_unknown_placeholder_raises_clearly(self):
        v = PromptVariant(
            name="t",
            system="S",
            user_template="hello {missing}",
        )
        with pytest.raises(PromptError, match="missing"):
            v.render_user()

    def test_malformed_template_raises_clearly(self):
        v = PromptVariant(name="t", system="S", user_template="hello {name")
        with pytest.raises(PromptError, match="malformed template"):
            v.render_user(name="x")

    def test_render_passes_through_format(self):
        v = PromptVariant(
            name="t",
            system="S",
            user_template="windows={windows_block}; scene={scene_description_block}",
        )
        result = v.render_user(windows_block="W", scene_description_block="S")
        assert result == "windows=W; scene=S"

    def test_extra_kwargs_ignored(self):
        # str.format ignores extras silently — formal contract.
        v = PromptVariant(name="t", system="S", user_template="just {a}")
        assert v.render_user(a="A", b="ignored") == "just A"
