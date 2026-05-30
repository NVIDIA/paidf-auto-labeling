# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from al_utils.schema.base import _clean_optional_str, _clean_required_str
from pydantic import BaseModel, ConfigDict, model_validator


class SampleInputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Primary input media path (video or image; local or remote: s3://, msc://, http(s)://)
    video_path: str
    # Advanced override: which media file VLM JSON / MCQ should consume for this sample.
    # If omitted, pipeline resolves tracking red-ID overlay -> SR output -> original input.
    vlm_video_path: Optional[str] = None
    # Optional sidecar artifacts (local or remote). If provided, they will be staged next to the input media.
    metadata_json_path: Optional[str] = None

    @model_validator(mode="after")
    def _require_video_path(self) -> "SampleInputsConfig":
        self.video_path = _clean_required_str(self.video_path, field="data[*].inputs.video_path")
        # Treat empty strings as unset for optional sidecars.
        for k in (
            "vlm_video_path",
            "metadata_json_path",
        ):
            setattr(self, k, _clean_optional_str(getattr(self, k), field=f"data[*].inputs.{k}"))
        return self


class SampleOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Per-sample output directory. This is the DAFT scene root:
    # - ``raw/<media_id>.<ext>`` — analyzed media (SR output if SR ran, else original input).
    #                              Local inputs use a symlink; remote-staged inputs are copied
    #                              so temp-file cleanup cannot break the scene.
    # - ``contextual/*.json``    — DAFT-compliant video/events/instances/objects.
    # - ``task/{mcq,bcq,open_qa}.json`` — DAFT-compliant task items.
    # - ``sidecars/*``           — diagnostic / mode-specific files (ignored by the DAFT validator).
    #
    # Accepts a local path or an MSC-backed remote URL (s3://, msc://, ...). Remote out_dir
    # means stages write to a local working copy and the CLI uploads it on success.
    out_dir: str
    # Optional log directory override. If omitted, runtime uses <out_dir>/logs.
    log_dir: Optional[str] = None
    # Optional path to save the effective/used config for this run (local or remote).
    # If omitted, CLI may write a default file under out_dir (e.g., out_dir/config.yaml).
    config_path: Optional[str] = None

    @model_validator(mode="after")
    def _validate_required_paths(self) -> "SampleOutputConfig":
        self.out_dir = _clean_required_str(self.out_dir, field="data[*].output.out_dir")

        for k in ("log_dir", "config_path"):
            setattr(self, k, _clean_optional_str(getattr(self, k), field=f"data[*].output.{k}"))
        return self


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: SampleInputsConfig
    output: SampleOutputConfig
