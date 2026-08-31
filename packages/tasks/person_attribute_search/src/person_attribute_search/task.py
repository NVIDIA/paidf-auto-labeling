# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Person Attribute Search assembly task.

This task is the orchestration "glue". By default it does **not** call any model:
it reads the artifacts produced by the existing ``visual_qa`` (structured
attributes via the PAS question bank) and ``captioning`` (natural caption)
services, then assembles and writes the PAS-native sidecars.

It has exactly one *optional* model op, gated by ``llm_query_generation`` and
mirroring the legacy ``generate_queries.py`` pipeline: a text LLM generates the
medium and hard retrieval queries from each person's attributes + caption
(``PROMPT_V3``), or hard queries only (``PROMPT_V3_HARD_ONLY``) when medium is
template-generated. With the flag off the task remains fully model-free. Easy
queries are always template-generated, as in the legacy pipeline. The task
writes the PAS-native sidecars:

- ``person_attribute_search/attributes.json`` and ``queries.json`` for legacy
  generation, or ``bundle_attributes.json`` and ``bundle_queries.json`` for
  bundle generation (with one entry per explicit attribute image)
- ``person_attribute_search/hitl.json`` for legacy generation, or
  ``bundle_hitl.json`` for explicit-attribute bundle generation

For the video flow the task runs a per-track (multi-person) flow and writes the
per-chunk documents:

- ``person_attribute_search/pas.json`` (``{chunk_id, pas: {n_people, people}, ...}``)
- ``person_attribute_search/chunk_queries.json`` (flat ``{queries: [{query}, ...]}``)

The per-track seam is assembled model-free from two generic upstream artifacts:
``detection_and_tracking`` crop bookkeeping (``tracks.json``) and ``visual_qa``
run once per track (each track's crops are one QA window stamped with
``track_id``). An explicit pre-assembled ``track_inputs.json`` seam is also
honored when present.

The structured sidecars above are always written (lossless round-trip). When
``emit_daft_contextual`` is set (default), the per-chunk flow *additionally*
mirrors the same content into two DAFT-enveloped contextual annotations so the
data is discoverable as first-class DAFT:

- ``contextual/person_attributes.json`` (envelope + the ``pas.json`` body)
- ``contextual/pas_queries.json`` (envelope + the ``chunk_queries.json`` body)

Both are registered DAFT ``contextual`` types. The ``pas_queries`` type is
deliberately namespaced (not the generic ``queries``) so it never collides with
other query sources across passes. PAS's open attribute vocabulary is preserved
verbatim inside the enveloped body; the DAFT copy adds structure, never rewrites
content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from core import (
    DataEntry,
    ScenePaths,
    ensure_scene_skeleton,
    read_json,
    resolve_sidecar,
    scene_context_for_entry,
    write_json,
)
from core.model_clients import EndpointClient, create_endpoint_client
from core.tasks import SequentialTask

from person_attribute_search.attributes import attributes_from_visual_qa_items
from person_attribute_search.bucket_queries import build_chunk_query_buckets
from person_attribute_search.bundle_queries import (
    BundleQueryParams,
    BundleQuerySetBuilder,
    resolve_bundle_prompt,
)
from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.export import benchmark, bundle, chunk_pas, hitl
from person_attribute_search.identity import make_person_key, person_key_to_string
from person_attribute_search.llm_queries import LlmQueryParams, LlmQuerySetBuilder
from person_attribute_search.outputs import (
    emit_anomaly_deliverable,
    emit_daft_contextual,
    record_state,
)
from person_attribute_search.queries import (
    QuerySet,
    assemble_query_set,
    collect_hard_queries,
    flatten_queries,
)
from person_attribute_search.schema import PersonAttributes, validate_attributes
from person_attribute_search.sources import (
    assemble_track_inputs_from_upstream,
    extract_caption,
    find_sidecar,
    load_attribute_image_sources,
    load_track_inputs,
    merge_attribute_json,
    resolve_attribute_items,
    string_list,
)
from person_attribute_search.tracks import (
    QuerySetBuilder,
    TrackRecord,
    build_people,
    track_record_from_mapping,
)


class PersonAttributeSearchTask(SequentialTask):
    """Assemble PAS attributes, tiered queries, and HITL preannotations."""

    def __init__(
        self,
        config: PersonAttributeSearchConfig | None = None,
        *,
        name: str | None = None,
        max_retries: int = 0,
    ) -> None:
        """
        Initialize the task with its assembly configuration.

        Args:
            config: Task configuration; defaults are used when ``None``.
            name: Task name for logging/state (defaults to
                ``"person_attribute_search"``).
            max_retries: Retry budget forwarded to ``SequentialTask``.
        """
        super().__init__(name=name or "person_attribute_search", max_retries=max_retries)
        self.config = config or PersonAttributeSearchConfig()
        self._llm_client: EndpointClient | None = self._build_llm_client()
        self._query_builder: QuerySetBuilder | None = self._build_query_builder(self._llm_client)
        self._bucket_client: EndpointClient | None = (
            self._llm_client if self.config.bucket_query_generation else None
        )

    def _build_llm_client(self) -> EndpointClient | None:
        """Build the shared text LLM client when any PAS query generator needs one."""
        if not (
            self.config.llm_query_generation
            or self.config.bucket_query_generation
            or self.config.bundle_query_generation
        ):
            return None
        return create_endpoint_client(
            provider=self.config.llm_provider,
            endpoint_url=self.config.llm_endpoint_url,
            model=self.config.llm_model,
            timeout_s=self.config.llm_timeout_s,
            retries=self.config.llm_retries,
            retry_backoff_s=self.config.llm_retry_backoff_s,
            logger=self.logger,
        )

    def _build_query_builder(self, client: EndpointClient | None) -> QuerySetBuilder | None:
        """
        Build the LLM tiered-query generator when configured, else ``None``.

        When ``bundle_query_generation`` is enabled it takes precedence: a single
        LLM call over the structured visual description produces all three tiers
        via the caller-supplied prompt (image-augmentation flow). Otherwise, when
        ``llm_query_generation`` is enabled this reproduces the legacy
        ``generate_queries.py`` model op: medium + hard queries (or hard only,
        when medium is template-generated) come from a text LLM. When both are
        disabled, the task stays fully model-free and templates/upstream items
        supply the queries.
        """
        if self.config.bundle_query_generation:
            if client is None:
                raise ValueError("bundle_query_generation requires an LLM endpoint client")
            prompt = resolve_bundle_prompt(
                prompt_text=self.config.query_prompt_text,
                prompt_file=self.config.query_prompt_file,
            )
            return BundleQuerySetBuilder(
                client,
                BundleQueryParams(
                    prompt=prompt,
                    max_tokens=self.config.bundle_max_tokens,
                    temperature=self.config.llm_temperature,
                    top_p=self.config.llm_top_p,
                    easy_cap=self.config.bundle_easy_count,
                    medium_cap=self.config.bundle_medium_count,
                    hard_cap=self.config.bundle_hard_count,
                    strict_count=self.config.bundle_query_count,
                ),
                response_retries=self.config.llm_response_retries,
                response_retry_backoff_s=self.config.llm_response_retry_backoff_s,
                logger=self.logger,
            )
        if not self.config.llm_query_generation:
            return None
        if client is None:
            raise ValueError("llm_query_generation requires an LLM endpoint client")
        params = LlmQueryParams(
            use_template_for_medium=self.config.use_template_for_medium,
            easy_count=self.config.easy_count,
            medium_count=self.config.medium_count,
            stop_words=tuple(self.config.stop_words),
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
        )
        return LlmQuerySetBuilder(
            client,
            params,
            response_retries=self.config.llm_response_retries,
            response_retry_backoff_s=self.config.llm_response_retry_backoff_s,
            logger=self.logger,
        )

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        """
        Assemble and write the PAS sidecars for one scene.

        Runs the per-track video flow when a ``track_inputs.json`` seam is
        present; otherwise the single-identity flow that sources attributes from
        ``visual_qa`` (or, when configured, a captioning ``parsed`` sidecar). The
        outcome is recorded in the scene pipeline state. The only model call is
        the optional LLM tiered-query generation (``llm_query_generation``).

        Args:
            data_entry: The scene's data entry; its ``data_path`` locates the
                scene directory and sidecars.

        Returns:
            The unmodified ``data_entry`` (results are written as sidecars).
        """
        if not self.config.enabled:
            self.logger.info("Person Attribute Search disabled; skipping.")
            return data_entry

        paths = ensure_scene_skeleton(data_entry.data_path)

        ctx = scene_context_for_entry(data_entry)
        if self.config.attribute_json:
            attribute_path = Path(self.config.attribute_json)
            self.logger.info("PAS attributes sourced from explicit attribute JSON")
            if self.config.bundle_query_generation:
                return self._run_attribute_image_bundle(
                    data_entry,
                    paths,
                    attribute_path=attribute_path,
                    fallback_chunk_id=ctx.media_id,
                )
            merged_source = merge_attribute_json(attribute_path)
            person_id = merged_source.person_id or ctx.media_id
            dataset = merged_source.dataset or self.config.dataset
            person_key = merged_source.person_key or person_key_to_string(
                make_person_key(dataset, person_id)
            )
            return self._run_single_identity(
                data_entry,
                paths,
                attributes=merged_source.attributes,
                hard_queries=[],
                person_key=person_key,
                person_id=person_id,
                dataset=dataset,
                image_ids=list(merged_source.image_ids) or [ctx.media_id],
            )

        track_inputs = load_track_inputs(paths, self.config.track_inputs_sidecar)
        if track_inputs is None:
            track_inputs = assemble_track_inputs_from_upstream(
                paths,
                chunk_id=ctx.media_id,
                tracks_sidecars=self.config.tracks_sidecars,
                visual_qa_window_sidecars=self.config.visual_qa_window_sidecars,
            )
        if track_inputs is not None:
            return self._run_per_track(data_entry, paths, track_inputs)

        items, source = resolve_attribute_items(
            paths,
            visual_qa_sidecars=self.config.visual_qa_item_sidecars,
            caption_attribute_sidecars=self.config.caption_attribute_sidecars,
        )
        if items is None:
            message = (
                "No usable PAS attribute source found; provide attribute_json or an upstream "
                "Visual QA sidecar"
            )
            self.logger.error(
                "%s for %s; checked visual_qa=%s captioning=%s",
                message,
                data_entry.data_path,
                ", ".join(self.config.visual_qa_item_sidecars),
                ", ".join(self.config.caption_attribute_sidecars) or "(disabled)",
            )
            record_state(data_entry, success=False, warnings=[message])
            return data_entry
        self.logger.info("PAS attributes sourced from %s", source)

        attributes = attributes_from_visual_qa_items(items)
        attributes = self._apply_caption_fallback(paths, attributes)
        person_id = ctx.media_id
        dataset = self.config.dataset
        person_key = person_key_to_string(make_person_key(dataset, person_id))
        return self._run_single_identity(
            data_entry,
            paths,
            attributes=attributes,
            hard_queries=collect_hard_queries(items, self.config.hard_query_ids),
            person_key=person_key,
            person_id=person_id,
            dataset=dataset,
            image_ids=[ctx.media_id],
        )

    def _run_attribute_image_bundle(
        self,
        data_entry: DataEntry,
        paths: ScenePaths,
        *,
        attribute_path: Path,
        fallback_chunk_id: str,
    ) -> DataEntry:
        """Generate and export one tiered query entry for every source image."""
        sources = load_attribute_image_sources(attribute_path)
        query_sets = [self._assemble_query_set(source.attributes, []) for source in sources]
        first_entry = sources[0].source_entry
        source_chunk_id = first_entry.get("chunk_id", first_entry.get("augmentation_id"))
        chunk_id = str(source_chunk_id) if source_chunk_id is not None else fallback_chunk_id
        attributes_document = bundle.build_image_attribute_bundle_document(
            chunk_id=chunk_id,
            sources=sources,
        )
        attributes_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_bundle_attributes_sidecar),
            attributes_document,
        )
        queries_document = bundle.build_image_bundle_document(
            chunk_id=chunk_id,
            sources=sources,
            query_sets=query_sets,
        )
        bundle_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_bundle_queries_sidecar),
            queries_document,
        )
        bundle_hitl_path: Path | None = None
        if self.config.write_hitl:
            bundle_hitl_path = write_json(
                resolve_sidecar(paths.sidecars_dir, self.config.output_bundle_hitl_sidecar),
                bundle.build_image_hitl_bundle_document(
                    chunk_id=chunk_id,
                    sources=sources,
                    query_sets=query_sets,
                    image_url_base=self.config.hitl_image_url_base,
                ),
            )
        record_state(
            data_entry,
            success=True,
            bundle_attributes_json=attributes_path,
            bundle_queries_json=bundle_path,
            bundle_hitl_json=bundle_hitl_path,
            n_people=len(sources),
        )
        self.logger.info(
            "PAS wrote image query bundle for %d image(s) to %s", len(sources), bundle_path
        )
        return data_entry

    def _run_single_identity(
        self,
        data_entry: DataEntry,
        paths: ScenePaths,
        *,
        attributes: PersonAttributes,
        hard_queries: list[str],
        person_key: str,
        person_id: str,
        dataset: str,
        image_ids: list[str],
    ) -> DataEntry:
        """Write legacy PAS artifacts for one normalized person identity."""
        warnings = validate_attributes(attributes)
        for warning in warnings:
            self.logger.warning("PAS attribute warning: %s", warning)

        query_set = self._assemble_query_set(attributes, hard_queries)
        images = {"all": image_ids}

        attributes_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_attributes_sidecar),
            benchmark.build_attribute_entry(
                person_key=person_key,
                person_id=person_id,
                dataset=dataset,
                images=images,
                attributes=attributes,
            ),
        )
        queries_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_queries_sidecar),
            benchmark.build_query_entry(
                person_key=person_key,
                person_id=person_id,
                dataset=dataset,
                attributes=attributes,
                query_set=query_set,
                flat_query_tiers=self.config.bundle_query_generation,
            ),
        )

        hitl_path: Path | None = None
        if self.config.write_hitl:
            image_url = f"{self.config.hitl_image_url_base}{image_ids[0]}"
            hitl_path = write_json(
                resolve_sidecar(paths.sidecars_dir, self.config.output_hitl_sidecar),
                hitl.build_preannotation_file(
                    image_url=image_url,
                    query_set=query_set,
                    natural_caption=attributes.natural_caption,
                ),
            )

        record_state(
            data_entry,
            success=True,
            attributes_json=attributes_path,
            queries_json=queries_path,
            hitl_json=hitl_path,
            warnings=warnings,
        )
        self.logger.info(
            "PAS wrote attributes/queries for person %s (easy=%d medium=%d hard=%d)",
            person_key,
            len(query_set.easy),
            len(query_set.medium),
            len(query_set.hard),
        )
        return data_entry

    def _run_per_track(
        self,
        data_entry: DataEntry,
        paths: ScenePaths,
        payload: dict[str, Any],
    ) -> DataEntry:
        """Per-track video flow: emit per-chunk PAS + queries documents."""
        raw_tracks = payload.get("tracks")
        if not isinstance(raw_tracks, list) or not raw_tracks:
            self.logger.info(
                "Per-track inputs for %s carried no tracks; skipping.",
                data_entry.data_path,
            )
            record_state(data_entry, success=False)
            return data_entry

        tracks: list[TrackRecord] = []
        for record in raw_tracks:
            if not isinstance(record, dict):
                continue
            try:
                tracks.append(track_record_from_mapping(record))
            except (ValueError, KeyError, TypeError) as exc:
                self.logger.warning("Skipping malformed PAS track record %r: %s", record, exc)
        if not tracks:
            self.logger.info(
                "Per-track inputs for %s held no valid track records; skipping.",
                data_entry.data_path,
            )
            record_state(data_entry, success=False)
            return data_entry
        people, query_sets = build_people(
            tracks,
            easy_count=self.config.easy_count,
            medium_count=self.config.medium_count,
            stop_words=self.config.stop_words,
            hard_query_ids=self.config.hard_query_ids,
            query_builder=self._query_builder,
        )
        optional_failures: list[str] = []

        ctx = scene_context_for_entry(data_entry)
        chunk_id = str(payload.get("chunk_id") or ctx.media_id)
        crop_root = str(payload.get("crop_root") or "")
        source_annotation = payload.get("source_annotation")
        source_annotation = str(source_annotation) if isinstance(source_annotation, str) else None
        anomaly_labels = string_list(payload.get("anomaly_labels"))
        extra_queries = string_list(payload.get("caption_queries"))

        queries = flatten_queries(
            query_sets,
            anomaly_labels=anomaly_labels,
            extra_queries=extra_queries,
        )
        query_buckets = build_chunk_query_buckets(
            self._bucket_client,
            paths=paths,
            people=people,
            flat_queries=queries,
            video_captions_sidecars=self.config.video_captions_sidecars,
            anomaly_items_sidecars=self.config.anomaly_items_sidecars,
            model_name=self.config.model_name,
            per_bucket=self.config.bucket_query_count,
            max_tokens=self.config.bucket_query_max_tokens,
            temperature=self.config.bucket_query_temperature,
            top_p=self.config.llm_top_p,
            response_retries=self.config.llm_response_retries,
            response_retry_backoff_s=self.config.llm_response_retry_backoff_s,
            logger=self.logger,
            optional_failures=optional_failures,
        )

        pas_document = chunk_pas.build_pas_document(
            chunk_id=chunk_id,
            people=people,
            crop_root=crop_root,
            source_annotation=source_annotation,
        )
        pas_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_pas_sidecar),
            pas_document,
        )
        queries_document = chunk_pas.build_queries_document(
            chunk_id=chunk_id,
            queries=queries,
            source_annotation=source_annotation,
            query_buckets=query_buckets,
        )
        queries_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_chunk_queries_sidecar),
            queries_document,
        )

        if self.config.bundle_query_generation:
            self._emit_bundle_outputs(
                paths, chunk_id=chunk_id, people=people, query_sets=query_sets
            )

        if self.config.emit_daft_contextual:
            failure = emit_daft_contextual(
                data_entry=data_entry,
                ctx=ctx,
                pas_document=pas_document,
                queries_document=queries_document,
                logger=self.logger,
            )
            if failure is not None:
                optional_failures.append(failure)

        if self.config.emit_contextual:
            failure = emit_anomaly_deliverable(data_entry=data_entry, logger=self.logger)
            if failure is not None:
                optional_failures.append(failure)

        record_state(
            data_entry,
            success=True,
            pas_json=pas_path,
            chunk_queries_json=queries_path,
            n_people=len(people),
            optional_failures=optional_failures,
        )
        self.logger.info(
            "PAS wrote per-chunk %s: n_people=%d queries=%d",
            chunk_id,
            len(people),
            len(queries),
        )
        return data_entry

    def _emit_bundle_outputs(
        self,
        paths: ScenePaths,
        *,
        chunk_id: str,
        people: list[dict[str, Any]],
        query_sets: list[QuerySet],
    ) -> None:
        """Write the aggregated + per-person tiered query-bundle documents."""
        aggregated, per_person = bundle.build_bundle_documents(
            chunk_id=chunk_id,
            people=people,
            query_sets=query_sets,
        )
        aggregated_path = write_json(
            resolve_sidecar(paths.sidecars_dir, self.config.output_bundle_queries_sidecar),
            aggregated,
        )
        if self.config.emit_bundle_per_person:
            for filename, document in per_person:
                write_json(
                    resolve_sidecar(
                        paths.sidecars_dir,
                        f"{self.config.bundle_per_person_subdir}/{filename}",
                    ),
                    document,
                )
        self.logger.info(
            "PAS wrote query bundle for %d person(s) to %s",
            len(per_person),
            aggregated_path,
        )

    def _apply_caption_fallback(
        self,
        paths: ScenePaths,
        attributes: PersonAttributes,
    ) -> PersonAttributes:
        """Fill ``natural_caption`` from a captioning sidecar when missing."""
        if attributes.natural_caption:
            return attributes
        sidecar = find_sidecar(paths, self.config.caption_sidecars)
        if sidecar is None:
            return attributes
        payload = read_json(sidecar)
        caption = extract_caption(payload)
        if caption:
            return attributes.model_copy(update={"natural_caption": caption})
        return attributes

    def _assemble_query_set(
        self,
        attributes: PersonAttributes,
        hard_queries: list[str],
    ) -> QuerySet:
        """Build a query set via the LLM generator when enabled, else templates."""
        if self._query_builder is not None:
            return self._query_builder(attributes, hard_queries)
        return assemble_query_set(
            attributes,
            hard_queries=hard_queries,
            easy_count=self.config.easy_count,
            medium_count=self.config.medium_count,
            stop_words=self.config.stop_words,
        )


__all__ = ["PersonAttributeSearchTask"]
