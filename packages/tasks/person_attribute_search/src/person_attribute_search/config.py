# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Person Attribute Search assembly task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PersonAttributeSearchConfig(BaseModel):
    """
    Configuration for ``PersonAttributeSearchTask``.

    The task is an *assembly* stage: it consumes artifacts produced by the
    upstream ``visual_qa`` and ``captioning`` services and writes the PAS-native
    sidecars (structured attributes, tiered queries, HITL preannotations).

    Attribute extraction and hard-query VLM calls happen upstream in
    ``visual_qa``. Difficulty-tiered query generation has one optional model op
    that mirrors the legacy ``generate_queries.py`` pipeline: when
    ``llm_query_generation`` is enabled the task calls a text LLM with the
    ``PROMPT_V3`` (or ``PROMPT_V3_HARD_ONLY``) prompt to produce medium and hard
    queries. With it disabled the task is fully model-free (medium/hard come from
    templates and upstream items). Easy queries are always template-generated, as
    in the legacy pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    # Optional explicit attribute artifact. When set, this source takes
    # precedence over every scene-local Visual QA or per-track sidecar. The
    # common PAS attribute envelopes accepted by the standalone query utility
    # are supported; multiple entries are merged as observations of one person.
    attribute_json: str | None = None

    # Upstream artifacts (relative to the scene ``sidecars/`` directory), tried
    # in order until one exists.
    visual_qa_item_sidecars: tuple[str, ...] = ("visual_qa/items.json",)
    caption_sidecars: tuple[str, ...] = ()

    # Optional additional attribute producer: a captioning sidecar whose
    # ``parsed`` JSON carries the structured attributes (when captioning is
    # driven by the PAS attribute prompt). Empty by default so ``visual_qa``
    # stays the authoritative source and behavior is unchanged.
    caption_attribute_sidecars: tuple[str, ...] = ()

    # Question-bank ids whose answers are treated as upstream "hard" queries.
    hard_query_ids: tuple[str, ...] = ("hard", "hard_1", "hard_2")

    # Per-track input seam (video flow). An explicit, pre-assembled seam. When
    # this sidecar exists the task runs the per-track (multi-person) flow and
    # emits the per-chunk PAS + queries documents directly.
    track_inputs_sidecar: str = "person_attribute_search/track_inputs.json"

    # Per-track flow without an explicit seam: assemble it model-free from the
    # generic upstream artifacts — ``detection_and_tracking`` crop bookkeeping
    # plus ``visual_qa`` run per track (each track's crops are one QA window,
    # stamped with ``track_id``). When both are present the task runs the
    # per-track flow; otherwise it falls back to the single-identity flow.
    tracks_sidecars: tuple[str, ...] = ("detection_and_tracking/tracks.json",)
    visual_qa_window_sidecars: tuple[str, ...] = ("visual_qa/windows.normalized.json",)

    # PAS-native output sidecars (relative to scene ``sidecars/``).
    output_attributes_sidecar: str = "person_attribute_search/attributes.json"
    output_queries_sidecar: str = "person_attribute_search/queries.json"
    output_hitl_sidecar: str = "person_attribute_search/hitl.json"

    # Per-chunk video output sidecars (relative to scene ``sidecars/``).
    output_pas_sidecar: str = "person_attribute_search/pas.json"
    output_chunk_queries_sidecar: str = "person_attribute_search/chunk_queries.json"

    # Chunk-level 3-bucket query generation (legacy ``query_generation.py``
    # parity). When enabled, the per-track flow makes one additional text-LLM
    # call grounded in the scene/dense captions + voted anomaly categories to
    # populate the ``Anomaly`` and ``Caption`` buckets of ``chunk_queries.json``
    # (the ``PAS`` bucket keeps the per-person queries). This requires pass-2
    # artifacts (captioning + anomaly visual_qa), so PAS must run after them.
    # Off by default so single-pass runs stay model-minimal and unchanged.
    bucket_query_generation: bool = False
    bucket_query_count: int = Field(default=5, ge=1)
    bucket_query_max_tokens: int = Field(default=800, ge=1)
    bucket_query_temperature: float = Field(default=0.5, ge=0.0)
    # Upstream caption + anomaly sources for bucket generation (first existing
    # wins), relative to the scene ``sidecars/`` directory.
    video_captions_sidecars: tuple[str, ...] = ("captioning/video_captions.json",)
    anomaly_items_sidecars: tuple[str, ...] = (
        "visual_qa_anomaly/items.json",
        "visual_qa/items.json",
    )

    # Query generation knobs.
    easy_count: int = Field(default=2, ge=0)
    medium_count: int = Field(default=2, ge=0)
    stop_words: tuple[str, ...] = ()

    # LLM tiered-query generation (legacy ``generate_queries.py`` parity).
    # When enabled the task calls a text LLM to generate the medium and hard
    # queries from each person's attributes + caption (``PROMPT_V3``). When
    # ``use_template_for_medium`` is set, the LLM produces hard queries only
    # (``PROMPT_V3_HARD_ONLY``) and medium queries are template-generated.
    llm_query_generation: bool = False
    use_template_for_medium: bool = False
    llm_provider: Literal["openai-compatible", "gemini"] = "openai-compatible"
    llm_endpoint_url: str | None = None
    llm_model: str = ""
    llm_max_tokens: int = Field(default=500, ge=1)
    llm_temperature: float = Field(default=0.7, ge=0.0)
    llm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    llm_timeout_s: float = Field(default=120.0, gt=0.0)
    llm_retries: int = Field(default=2, ge=0)
    llm_retry_backoff_s: float = Field(default=2.0, ge=0.0)
    # Separate retry budget for successful endpoint responses that are empty,
    # malformed, or do not satisfy the query-generation contract.
    llm_response_retries: int = Field(default=2, ge=0)
    llm_response_retry_backoff_s: float = Field(default=2.0, ge=0.0)

    # Single-call tiered "query bundle" generation (PAS image-augmentation flow).
    # When enabled, one LLM call over each person's structured visual description
    # produces all three difficulty tiers in the
    # ``{"queries": {"easy", "medium", "hard"}}`` shape, replacing the per-tier
    # ``llm_query_generation`` path (which builds ``easy`` from templates). The
    # prompt is fully caller-supplied via ``query_prompt_text``/``query_prompt_file``
    # so the query specification can change without code edits. Off by default so
    # the video flow is unchanged.
    bundle_query_generation: bool = False
    query_prompt_text: str | None = None
    query_prompt_file: str | None = None
    # Keep well below common 16k-context LLM windows so prompt + completion fit.
    bundle_max_tokens: int = Field(default=2048, ge=1)
    # Required number of distinct queries in every tier (``0`` = tolerant).
    # Use this to enforce the exact per-tier count requested by the bundle prompt.
    bundle_query_count: int = Field(default=0, ge=0)
    # Optional per-tier caps (``0`` = keep whatever the prompt returns).
    bundle_easy_count: int = Field(default=0, ge=0)
    bundle_medium_count: int = Field(default=0, ge=0)
    bundle_hard_count: int = Field(default=0, ge=0)
    # Aggregated bundle output. Explicit attribute-image inputs receive one entry
    # per image (duplicate person keys are disambiguated by image id). Per-track
    # inputs are keyed by ``<person_key>`` and may additionally emit one document
    # per person under ``bundle_per_person_subdir``.
    output_bundle_attributes_sidecar: str = "person_attribute_search/bundle_attributes.json"
    output_bundle_queries_sidecar: str = "person_attribute_search/bundle_queries.json"
    output_bundle_hitl_sidecar: str = "person_attribute_search/bundle_hitl.json"
    emit_bundle_per_person: bool = True
    bundle_per_person_subdir: str = "person_attribute_search/queries"

    # Identity metadata used when the scene does not carry a dataset/person id.
    dataset: str = "upa"

    # Export toggles.
    write_hitl: bool = True
    hitl_image_url_base: str = ""

    # Merged anomaly deliverable. When enabled, the per-track flow folds in the
    # annotation-schema adapter and writes the merged pass1+pass2
    # ``sidecars/person_attribute_search/pas_anomaly.json`` record right after the
    # PAS sidecars. Because PAS is the terminal pass-2 stage, this makes the
    # merged export a natural part of pass 2 instead of a separate, easily-missed
    # step. The name is kept for cookbook compatibility. Off by default so
    # single-pass/non-legacy runs are unchanged.
    emit_contextual: bool = False

    # DAFT contextual dual-write. When enabled (default), the per-chunk flow
    # mirrors ``pas.json`` / ``chunk_queries.json`` into DAFT-enveloped
    # ``contextual/person_attributes.json`` and ``contextual/pas_queries.json``
    # so the same content is discoverable as first-class DAFT contextual
    # annotations. The original sidecars are always kept (lossless round-trip);
    # this only adds the enveloped copies. The ``pas_queries`` type is namespaced
    # (not the generic ``queries``) to avoid collisions with other query sources.
    emit_daft_contextual: bool = True

    # Metadata stamped into benchmark envelopes.
    model_name: str = ""


__all__ = ["PersonAttributeSearchConfig"]
