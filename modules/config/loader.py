# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import multistorageclient as msc
import yaml
from nvcf_msc_utils import _download_http_file, convert_to_msc_url, http_timeout_seconds, is_http_path, is_remote_path
from omegaconf import OmegaConf

# Keys whose CLI override values are path-like; relative paths are resolved against CWD.
# pipeline + data[*] + stages that read prior outputs (tracking/vlm_json/mcq_generation).
_PATH_KEY_SUFFIXES = (
    ".model_cache_path",
    ".scene_prompt_file",
    ".events_prompt_file",
    "_prompt_file",
    ".output_file",
    ".inputs.video_path",
    ".inputs.vlm_video_path",
    ".inputs.metadata_json_path",
    ".output.out_dir",
    ".output.log_dir",
    ".output.config_path",
)


def _strip_wrapping_quotes(value: str) -> str:
    v = str(value)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
        return v[1:-1]
    return v


def _max_data_index(overrides: List[str]) -> int:
    """Return the highest ``data.N.*`` index found in overrides, or -1 if none."""
    max_idx = -1
    for item in overrides:
        m = re.match(r"^data\.(\d+)\.", item)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def _normalize_dotlist_overrides(overrides: List[str]) -> List[str]:
    """Resolve relative path overrides against CWD so they are not interpreted relative to the config file."""
    out: List[str] = []
    cwd = Path.cwd()
    for item in overrides:
        if "=" not in item:
            out.append(item)
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip()

        # Don't touch interpolations or empty values.
        if not value or "${" in value:
            out.append(item)
            continue

        if key.endswith(_PATH_KEY_SUFFIXES):
            v = _strip_wrapping_quotes(value)
            if not is_remote_path(v):
                p = Path(v).expanduser()
                if not p.is_absolute():
                    p = (cwd / p).resolve()
                out.append(f"{key}={p}")
                continue

        out.append(item)
    return out


def load_config_with_overrides(
    config_path: str, overrides: List[str], logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Load YAML config and apply dotlist overrides (key=value).
    Supports interpolation via OmegaConf resolve.
    """
    remote = is_remote_path(config_path)
    cfg_path: Path
    tmp_path: Optional[Path] = None

    if remote:
        # Download remote config into a temp file, then treat it like a normal YAML config.
        suffix = ".yaml"
        path_after_scheme = config_path.split("://", 1)[-1].split("?")[0]
        segments = path_after_scheme.strip("/").split("/")
        if segments and "." in segments[-1]:
            suffix = Path(segments[-1]).suffix or suffix

        fd, p = tempfile.mkstemp(prefix="auto_labeling_config_", suffix=suffix)
        os.close(fd)
        tmp_path = Path(p)

        try:
            # path_mapping can rewrite public-looking HTTPS storage URLs to MSC backends.
            # Unmapped HTTP(S) config files are downloaded directly so public URLs do not
            # require MSC configuration.
            resolved_config_path = convert_to_msc_url(config_path)
            if is_http_path(resolved_config_path):
                _download_http_file(resolved_config_path, str(tmp_path), timeout=http_timeout_seconds())
            else:
                msc.download_file(resolved_config_path, str(tmp_path))
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download remote config from {config_path}; set MULTISTORAGECLIENT_CONFIGURATION (JSON) for storage access when using MSC-backed paths."
            ) from e

        cfg_path = tmp_path
    else:
        cfg_path = Path(config_path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        conf = OmegaConf.create(base)
        OmegaConf.set_struct(conf, False)

        if overrides:
            overrides = _normalize_dotlist_overrides(list(overrides))
            if logger is not None:
                logger.info(f"Applying CLI overrides: {overrides}")
            # Auto-extend data[] list if overrides reference indices beyond what the
            # base config defines (e.g. data.1.*, data.2.* when base only has data[0]).
            max_idx = _max_data_index(overrides)
            if max_idx >= 0 and "data" in conf and OmegaConf.is_list(conf.data):
                while max_idx >= len(conf.data):
                    conf.data.append(copy.deepcopy(conf.data[0]))
            conf.merge_with_dotlist(overrides)

        resolved = OmegaConf.to_container(conf, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Config root must be a mapping/dict")
        # For remote configs, there is no meaningful "config directory" on disk. Use CWD so
        # relative paths are interpreted in a stable, user-controlled way.
        config_dir = Path.cwd() if remote else cfg_path.parent
        return resolved, config_dir
    except Exception as e:
        raise RuntimeError(f"Failed to read config file {cfg_path} or apply overrides: {e}") from e
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
