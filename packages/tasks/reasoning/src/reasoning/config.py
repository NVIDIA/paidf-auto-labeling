# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config schema for opt-in DAFT-export stages.

Deterministic captioning and visual-QA pivots are service-owned and live in
``core.formats.daft``. This package owns DAFT-export stages that make
additional inference calls.

This module holds knobs for stages that make additional inference calls
(MSTED, reasoning enrichment, temporal localization, the open-ended QA
family, causal linkage) and so are opt-in. Each sub-block carries
enough info to:

- Decide whether to run (``enabled`` flag).
- Pick a prompt variant from the prompt registry.
- Layer additional prompt directories on top of the bundled defaults
  (the use-case-agnostic extension hook the user asked for).
- Configure the LLM call (max_tokens, temperature, etc.); endpoints are
  resolved by the service through CLI overrides and ``EndpointResolver``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DEFAULT_DESCRIPTION_KEYS: tuple[str, ...] = (
    "enhanced_caption",
    "description",
    "caption",
    "summary",
)


class _BaseLLMStageConfig(BaseModel):
    """Shared knob block for opt-in, LLM-driven DAFT-export stages.

    Captures the fields every stage in this module needs (``enabled``
    flag, prompt selection, LLM call tuning, structured-output mode,
    description-key fallbacks). Family-specific configs below add
    their input-bank fields on top of this without re-declaring the
    LLM knobs.

    Subclasses typically override ``prompt_variant`` (each stage has
    its own bundled default) and may override numeric defaults
    (``max_tokens``, ``timeout``) when their prompts demand a
    different budget. Everything else stays the same so that adding
    a new opt-in stage is a one-class affair.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    prompt_variant: str
    prompt_dir: str | None = None

    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=600, ge=1)

    structured_output: Literal["auto", "nim", "openai", "off"] = "auto"

    seed: int | None = None
    retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=5.0, ge=0.0)

    description_keys: tuple[str, ...] = _DEFAULT_DESCRIPTION_KEYS


class EventsConfig(_BaseLLMStageConfig):
    """Config for the LLM-aggregated events stage (``contextual/events.json``).

    When enabled, this stage **replaces** the events.json that the upstream
    ``vlm_json`` whole-clip pass writes. The aggregator reads per-window
    dense captions from ``sidecars/metadata_chunk.json`` and the instances
    catalogue from ``contextual/instances.json``, then asks an LLM to
    emit a list of noteworthy temporal events grounded in that catalogue.

    The stage is opt-in (``enabled: false`` by default) so existing
    pipelines aren't surprised by new LLM traffic — and so the legacy
    whole-clip events.json remains the default behavior until a pipeline
    explicitly opts in.

    Domain specialization is the same hook as MSTED: swap
    ``prompt_variant`` (the bundled default is ``events_from_chunks``,
    intentionally generic) and/or drop a YAML in ``prompt_dir``. **No
    prompt text or category vocabulary is hard-coded in Python.**

    Cost note: roughly one LLM call per scene.
    """

    prompt_variant: str = "events_from_chunks"
    # Events aggregator output is verbose (multi-event with optional
    # per-event placeholders); keep a 4096-token budget rather than the
    # smaller base default so a long aftermath sequence doesn't truncate.
    max_tokens: int = Field(default=4096, ge=1)

    max_events: int = Field(
        default=200,
        ge=1,
        description=(
            "Hard cap on the number of events accepted from the LLM. "
            "Generous on purpose — a long video legitimately has many "
            "events — but capped so a runaway hallucination doesn't "
            "write a 100k-item file."
        ),
    )


class MstedConfig(_BaseLLMStageConfig):
    """Config for the LLM-aggregated MSTED stage (``contextual/msted.json``).

    MSTED is the Multi-Scale Spatio-Temporal Event Description: a single
    per-scene file that aggregates per-window VLM captions into a
    structured event characterization. Production cost is roughly one
    LLM call per scene; this stage is opt-in (``enabled: false`` by
    default) so existing pipelines aren't surprised by new LLM traffic.

    Domain specialization is a single config knob (``prompt_variant``)
    plus a YAML drop-in under ``prompt_dir``: the bundled
    ``msted_default`` variant is intentionally generic, and any
    ``<variant_name>.yaml`` placed in ``prompt_dir`` overrides the
    bundled variant of the same name. **No prompt text is hard-coded
    in Python; switching domain never requires a code change.**
    """

    prompt_variant: str = "msted_default"
    # MSTED prompts produce dense per-segment output; keep the historical
    # 4096-token budget rather than the smaller base default.
    max_tokens: int = Field(default=4096, ge=1)

    max_segments: int = Field(default=10_000, ge=1)


ReasoningTarget = Literal[
    "scene_description",
    "video_summarization",
    "temporal_description",
    "open_qa",
    "mcq_openended",
    "bcq_openended",
    "causal_linkage",
]


def _default_reasoning_targets() -> list[ReasoningTarget]:
    return ["scene_description", "video_summarization"]


class ReasoningConfig(BaseModel):
    """Config for the LLM reasoning-enrichment pass.

    When enabled, the pipeline calls the LLM once per task item in the
    selected ``targets`` to fill the optional ``reasoning`` field. Items
    are written with reasoning attached when the call succeeds and
    without when it fails — the reasoning enrichment is best-effort and
    never blocks the underlying task file.

    Cost note: ``len(targets) * len(items_per_target)`` LLM calls per
    scene. ``scene_description`` and ``video_summarization`` are 1 item
    each (cheap). ``temporal_description`` is N items (one per window),
    so enable that target only when you actually want per-window
    reasoning at proportional LLM cost.

    Use-case agnosticism is the same hook as MSTED: prompt comes from a
    YAML in the registry, swap variant via ``prompt_variant``, ship
    domain-specific YAMLs in ``prompt_dir``. **No prompt text is hard-coded.**
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    prompt_variant: str = "reasoning_default"

    prompt_dir: str | None = None

    targets: list[ReasoningTarget] = Field(default_factory=_default_reasoning_targets)

    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=120, ge=1)

    seed: int | None = None
    retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=5.0, ge=0.0)

    # Reasoning traces are now structured (observation -> inference ->
    # conclusion) and may span one to two short paragraphs; give them
    # headroom so a legitimate multi-step trace isn't truncated.
    max_chars: int = Field(default=4_000, ge=1)

    include_context: bool = True

    @field_validator("targets")
    @classmethod
    def _dedupe_targets(cls, v: list[ReasoningTarget]) -> list[ReasoningTarget]:
        seen: set[str] = set()
        out: list[ReasoningTarget] = []
        for t in v:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if not out:
            raise ValueError("reasoning.reasoning.targets must be non-empty")
        return out


class TemporalLocalizationConfig(_BaseLLMStageConfig):
    """Config for the LLM temporal-localization stage (``task/temporal_localization.json``).

    The query bank is **caller-supplied data**. There are three input
    modes (any combination — they're concatenated and deduped):

    - ``queries: ["..."]`` — inline list in the pipeline config. Use
      this for small, hand-curated query sets.
    - ``query_file: <path>`` — YAML or JSON file with one of the
      supported shapes (bare list, ``{"queries": [...]}``,
      MCQ-style ``{"questions": [{"text": "..."}, ...]}``). Use this
      to share a query bank across pipelines, or to reuse the existing
      MCQ ``window_metadata_extraction.question_bank_file``.
    - Both: inline + file are concatenated.

    When ``enabled=true`` but no queries resolve, the stage logs a
    warning and skips writing — the file is opt-in twice (feature flag
    *and* non-empty query bank), so existing pipelines aren't surprised
    by new LLM traffic.

    Domain specialization works the same way as MSTED: swap
    ``prompt_variant`` and/or drop a YAML in ``prompt_dir``. **No
    prompt text or query is hard-coded in Python.**
    """

    prompt_variant: str = "temporal_localization_default"
    # Per-query grounding output is verbose; keep the historical 4096
    # budget instead of the smaller base default.
    max_tokens: int = Field(default=4096, ge=1)

    queries: list[str] = Field(default_factory=list)
    query_file: str | None = None

    @model_validator(mode="after")
    def _require_query_source_when_enabled(self) -> TemporalLocalizationConfig:
        if self.enabled and not self.queries and not self.query_file:
            raise ValueError(
                "reasoning.temporal_localization.enabled=true requires either "
                "'queries' (inline list) or 'query_file' (path); both are empty"
            )
        return self


class OpenQaConfig(_BaseLLMStageConfig):
    """Config for the LLM open-ended QA stage (``task/open_qa.json``).

    Two input modes (any combination — concatenated and deduped):

    - ``questions: ["..."]`` — inline list of free-form question strings.
    - ``question_file: <path>`` — YAML/JSON file with one of the
      supported shapes (bare list, ``{"questions": [...]}``,
      MCQ-style ``{"questions": [{"text": "..."}, ...]}``); the same
      shapes used by ``temporal_localization`` so a single bank can
      drive multiple stages.

    When ``enabled=true`` but no questions resolve, the stage logs a
    warning and skips writing — the file is opt-in twice (feature flag
    *and* non-empty bank), so existing pipelines aren't surprised by
    new LLM traffic.

    Domain specialization is the same hook as MSTED: swap
    ``prompt_variant`` and/or drop a YAML in ``prompt_dir``. **No
    prompt text or question is hard-coded in Python.**
    """

    prompt_variant: str = "open_qa_default"
    # Open-ended answers now carry a multi-paragraph Observation ->
    # Inference -> Conclusion reasoning trace per item; keep the verbose
    # 4096-token budget so a multi-question bank doesn't truncate.
    max_tokens: int = Field(default=4096, ge=1)

    questions: list[str] = Field(default_factory=list)
    question_file: str | None = None

    @model_validator(mode="after")
    def _require_question_source_when_enabled(self) -> OpenQaConfig:
        if self.enabled and not self.questions and not self.question_file:
            raise ValueError(
                "reasoning.open_qa.enabled=true requires either 'questions' "
                "(inline list) or 'question_file' (path); both are empty"
            )
        return self


class McqOpenendedItem(BaseModel):
    """One MCQ open-ended bank entry: a question plus its letter-keyed options.

    The ``options`` mapping is optional in the DAFT schema (the LLM
    can embed choices inline in the question prose); however the
    bundled prompt assumes options are passed via this field, and the
    converter enforces ``minProperties: 2`` when ``options`` is
    present. Authors who want inline-options style should pass
    ``options=None`` and embed the choices in the question text."""

    model_config = ConfigDict(extra="forbid")

    question: str
    options: dict[str, str] | None = None


class McqOpenendedConfig(_BaseLLMStageConfig):
    """Config for the LLM open-ended MCQ stage (``task/mcq_openended.json``).

    Same input-bank pattern as :class:`OpenQaConfig` but each entry
    optionally carries an ``options`` letter-keyed dict (e.g.
    ``{"A": "passenger car", "B": "truck"}``). The bank can be loaded
    from a file as well; see :func:`reasoning.qa.llm.load_qa_bank`
    for accepted file shapes.
    """

    prompt_variant: str = "mcq_openended_default"
    # Open-ended explanations carry a multi-step reasoning trace per item;
    # keep the verbose 4096-token budget so a multi-item bank doesn't
    # truncate mid-JSON.
    max_tokens: int = Field(default=4096, ge=1)

    items: list[McqOpenendedItem] = Field(default_factory=list)
    item_file: str | None = None

    @model_validator(mode="after")
    def _require_item_source_when_enabled(self) -> McqOpenendedConfig:
        if self.enabled and not self.items and not self.item_file:
            raise ValueError(
                "reasoning.mcq_openended.enabled=true requires either 'items' "
                "(inline list) or 'item_file' (path); both are empty"
            )
        return self


class BcqOpenendedConfig(_BaseLLMStageConfig):
    """Config for the LLM open-ended BCQ stage (``task/bcq_openended.json``).

    Yes/No question family. Same input-bank pattern as
    :class:`OpenQaConfig`; entries are plain strings (no options).
    """

    prompt_variant: str = "bcq_openended_default"
    # Yes/No explanations carry a multi-step reasoning trace per item;
    # keep the verbose 4096-token budget so a multi-question bank doesn't
    # truncate mid-JSON.
    max_tokens: int = Field(default=4096, ge=1)

    questions: list[str] = Field(default_factory=list)
    question_file: str | None = None

    @model_validator(mode="after")
    def _require_question_source_when_enabled(self) -> BcqOpenendedConfig:
        if self.enabled and not self.questions and not self.question_file:
            raise ValueError(
                "reasoning.bcq_openended.enabled=true requires either "
                "'questions' (inline list) or 'question_file' (path); both are empty"
            )
        return self


class CausalPair(BaseModel):
    """One explicit causal pair: ``{t1, t2, question?, video_type?}``.

    Used by :class:`CausalLinkageConfig` in ``mode="explicit"``. ``t1``
    and ``t2`` may be either numeric seconds or DAFT timecode strings;
    the adapter normalizes both to canonical timecodes before calling
    the LLM. ``question`` defaults to the bundled neutral template
    when omitted; ``video_type``, when set, is preserved through the
    LLM call into the final DAFT item."""

    model_config = ConfigDict(extra="forbid")

    t1: Any
    t2: Any
    question: str | None = None
    video_type: Literal["anomaly", "normal"] | None = None


class CausalLinkageConfig(_BaseLLMStageConfig):
    """Config for the LLM causal-linkage stage (``task/causal_linkage.json``).

    Two pair-source modes (selected by ``mode``):

    - ``"auto_from_events"`` (default): derive (t1, t2) pairs from
      ``contextual/events.json`` (consecutive event ``start_time``
      pairing, with the scene-end as the trailing anchor). Capped at
      ``max_pairs_auto`` to bound LLM cost on events-dense clips.
      Zero config — works on any clip the pipeline already produces
      events.json for.
    - ``"explicit"``: caller supplies ``pairs`` (inline) and/or
      ``pair_file`` (YAML/JSON). Each entry is a :class:`CausalPair`.

    When ``enabled=true`` and ``mode="explicit"`` but no pairs
    resolve, the stage raises at config-validation time. When
    ``mode="auto_from_events"`` and no events.json exists, the stage
    logs a warning and skips writing (best-effort).

    ``default_video_type``, when set, is attached to every
    auto-derived pair so the LLM preserves the classification through
    its answer items."""

    prompt_variant: str = "causal_linkage_default"
    # Each pair now yields a multi-paragraph Observation -> Logical Bridge
    # -> Conclusion reasoning trace; keep the verbose 4096-token budget so
    # an events-dense clip's pair set doesn't truncate mid-JSON.
    max_tokens: int = Field(default=4096, ge=1)

    mode: Literal["auto_from_events", "explicit"] = "auto_from_events"

    pairs: list[CausalPair] = Field(default_factory=list)
    pair_file: str | None = None

    max_pairs_auto: int = Field(default=8, ge=1)
    default_video_type: Literal["anomaly", "normal"] | None = None

    @model_validator(mode="after")
    def _require_pair_source_when_explicit(self) -> CausalLinkageConfig:
        if self.enabled and self.mode == "explicit" and not self.pairs and not self.pair_file:
            raise ValueError(
                "reasoning.causal_linkage.enabled=true with mode='explicit' "
                "requires either 'pairs' (inline list) or 'pair_file' (path); "
                "both are empty"
            )
        return self


class AnomalyConfig(_BaseLLMStageConfig):
    """Config for the LLM anomaly-classification stage.

    Unlike the other stages in this module, anomaly output is **not** a
    DAFT contextual/task type (the DAFT v3 type set is closed). It is a
    diagnostic *sidecar* at ``sidecars/reasoning/anomaly.json`` carrying:

    - ``classification`` — ``"anomaly"`` or ``"normal"``.
    - ``reasoning`` — a step-by-step trace (observation -> inference ->
      verdict) justifying the classification.
    - ``root_cause`` / ``consequence`` — present for anomalies.
    - ``highlight`` — when :attr:`localize_highlight` is set and the clip
      is an anomaly, the ``{start, end, description}`` window where the
      anomaly is most visible. This doubles as a **re-perception
      request**: a downstream captioning (VLM) pass can re-caption just
      that window at higher fidelity. Localizing the moment is text
      reasoning over captions and belongs here; the VLM re-caption itself
      is the captioning service's responsibility (this stage is LLM-only
      by design).

    The verdict also feeds ``causal_linkage`` auto-tagging: when set, the
    classification is used as the default ``video_type`` for auto-derived
    causal pairs (unless the causal config already pins one).

    Domain specialization is the same hook as MSTED: swap
    ``prompt_variant`` and/or drop a YAML in ``prompt_dir``. **No prompt
    text or anomaly vocabulary is hard-coded in Python.**

    Cost note: one LLM call per scene.
    """

    prompt_variant: str = "anomaly_classify_default"
    # Classification plus a localized highlight window is a compact
    # payload; the smaller base budget is plenty and keeps the per-scene
    # cost low.
    max_tokens: int = Field(default=1024, ge=1)

    localize_highlight: bool = Field(
        default=True,
        description=(
            "When true, anomalies also carry a localized {start, end, "
            "description} highlight window in the sidecar (the "
            "re-perception request). Set false to classify only."
        ),
    )

    include_person_attributes: bool = Field(
        default=True,
        description=(
            "When true and a person-attribute-search pass wrote "
            "sidecars/person_attribute_search/pas.json, fold the grounded "
            "per-person attributes (action, potential-anomaly flag/type, "
            "caption) into the anomaly prompt as corroborating evidence. "
            "Absent PAS output, this is a no-op. Set false to classify from "
            "captions only."
        ),
    )


class DaftExportConfig(BaseModel):
    """Top-level config for opt-in DAFT-export stages.

    Pure pivots run without a config block; this section only carries knobs
    for stages that need extra inference or a prompt registry. Adding a new
    opt-in stage means adding a sibling field here with the same shape:
    ``enabled`` + ``prompt_variant`` + ``prompt_dir`` + LLM call knobs.
    """

    model_config = ConfigDict(extra="forbid")

    events: EventsConfig | None = None
    msted: MstedConfig | None = None
    anomaly: AnomalyConfig | None = None
    reasoning: ReasoningConfig | None = None
    temporal_localization: TemporalLocalizationConfig | None = None
    open_qa: OpenQaConfig | None = None
    mcq_openended: McqOpenendedConfig | None = None
    bcq_openended: BcqOpenendedConfig | None = None
    causal_linkage: CausalLinkageConfig | None = None


__all__ = [
    "AnomalyConfig",
    "BcqOpenendedConfig",
    "CausalLinkageConfig",
    "CausalPair",
    "DaftExportConfig",
    "EventsConfig",
    "McqOpenendedConfig",
    "McqOpenendedItem",
    "MstedConfig",
    "OpenQaConfig",
    "ReasoningConfig",
    "ReasoningTarget",
    "TemporalLocalizationConfig",
]
