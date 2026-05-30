# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM JSON generator factory.

Raises ``ValueError`` on invalid config; never returns ``None``
(the call site initialises ``vlm_json_gen = None`` and only calls this when
the section is enabled).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from vlm_json.base import BaseVlmJsonGenerator
from vlm_json.vlm_json_generator import VlmJsonGenerator


def create_vlm_json_generator(
    config: PipelineConfig,
    resolver: EndpointResolver,
    logger: logging.Logger,
    config_dir: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> BaseVlmJsonGenerator:
    """Create the VLM JSON generator from configuration.

    Args:
        config:     Full pipeline config (Pydantic model).
        resolver:   Pre-constructed endpoint resolver.
        logger:     Logger instance.
        config_dir: Directory of the config file; used to resolve relative
                    prompt file paths.
        repo_root:   Repository root used to resolve shipped cookbook prompt assets.

    Returns:
        Configured ``BaseVlmJsonGenerator`` instance.

    Raises:
        ValueError: If ``vlm_json`` section is absent or VLM endpoint is
                    not configured.
    """
    if config.vlm_json is None:
        raise ValueError("'vlm_json' section is required to create a VLM JSON generator")

    model = str(config.vlm_json.model)
    if model == "vlm":
        return VlmJsonGenerator(
            config=config,
            resolver=resolver,
            logger=logger,
            config_dir=config_dir,
            repo_root=repo_root,
        )

    raise ValueError(f"Unknown vlm_json.model: {model!r}")
