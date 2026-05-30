# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_cache_path: Optional[str] = None
    gpu_ids: Union[str, int] = "all"
    # When true, SR uses all GPUs in gpu_ids (sp_size = len(gpu_ids)).
    # When false (default), SR uses only the first GPU in gpu_ids (single-GPU mode).
    use_multi_gpu: bool = False
    # Empty-output / reported-failure behavior for fallback-capable runtime stages (see pipeline.py):
    # - warn: warn + continue when downstream stages can still run
    # - fail: hard-fail on empty outputs / reported failures
    empty_output_policy: Optional[Literal["warn", "fail"]] = None

    # Run `tao-daft validate` on each completed scene directory. Best-effort:
    # the hook is a no-op when the `tao-daft` CLI isn't on PATH (the
    # `nvidia-tao-daft` package isn't shipped in the container by default).
    # Validator output is logged; it never fails the pipeline.
    daft_validate: bool = True

    # NOTE: Per-sample working directory is configured via data[*].output.out_dir.
