# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCQ generation factory.

Dispatches on ``config.mcq_generation.mode`` to the appropriate generator class.
Raises ``ValueError`` on unknown mode or invalid config; never returns ``None``
(the call site initialises ``mcq_gen = None`` and only calls this when the
section is enabled).
"""

from __future__ import annotations

import logging
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from mcq_generation.base import BaseMCQGenerator
from mcq_generation.metadata_llm import MetadataLlmGenerator
from mcq_generation.question_driven_vlm_llm import QuestionDrivenVlmLlmGenerator
from mcq_generation.window_direct_vlm import WindowDirectVlmGenerator
from mcq_generation.window_vlm_llm import WindowVlmLlmGenerator


def create_mcq_generator(
    config: PipelineConfig,
    resolver: EndpointResolver,
    logger: logging.Logger,
    config_dir: Optional[str] = None,
) -> BaseMCQGenerator:
    """Create the appropriate MCQ generator from configuration.

    Dispatches on ``config.mcq_generation.mode``:
    - ``window-vlm-llm``            → :class:`WindowVlmLlmGenerator`
    - ``window-direct-vlm``         → :class:`WindowDirectVlmGenerator`
    - ``question-driven-vlm-llm``   → :class:`QuestionDrivenVlmLlmGenerator`
    - ``metadata-llm``              → :class:`MetadataLlmGenerator`

    Args:
        config:     Full pipeline config (Pydantic model).
        resolver:   Pre-constructed endpoint resolver.
        logger:     Logger instance.
        config_dir: Directory of the config file; used to resolve relative
                    prompt file paths.

    Returns:
        Configured ``BaseMCQGenerator`` instance.

    Raises:
        ValueError: If ``mcq_generation`` section is absent, mode is unknown,
                    or required endpoints / prompt files are missing.
    """
    mcq_cfg = config.mcq_generation
    if mcq_cfg is None:
        raise ValueError("'mcq_generation' section is required to create an MCQ generator")

    mode = str(mcq_cfg.mode)

    if mode == "window-vlm-llm":
        _require_window_cfg(mode, config)
        return WindowVlmLlmGenerator(config=config, resolver=resolver, logger=logger, config_dir=config_dir)

    if mode == "window-direct-vlm":
        _require_window_cfg(mode, config)
        return WindowDirectVlmGenerator(config=config, resolver=resolver, logger=logger, config_dir=config_dir)

    if mode == "question-driven-vlm-llm":
        _require_window_cfg(mode, config)
        return QuestionDrivenVlmLlmGenerator(config=config, resolver=resolver, logger=logger, config_dir=config_dir)

    if mode == "metadata-llm":
        _require_window_cfg(mode, config)
        return MetadataLlmGenerator(config=config, resolver=resolver, logger=logger, config_dir=config_dir)

    valid = "window-vlm-llm, window-direct-vlm, question-driven-vlm-llm, metadata-llm"
    raise ValueError(f"Unknown mcq_generation.mode: {mode!r}. Valid modes: {valid}")


def _require_window_cfg(mode: str, config: PipelineConfig) -> None:
    if not config.mcq_generation or not config.mcq_generation.window_metadata_extraction:
        raise ValueError(
            f"mcq_generation.mode={mode!r} requires mcq_generation.window_metadata_extraction to be configured"
        )
