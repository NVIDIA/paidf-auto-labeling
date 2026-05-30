# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NVCF and MSC (Multi-Storage Client) Utility Functions.

- MSC config: MULTISTORAGECLIENT_CONFIGURATION (JSON string) is written to /tmp/msc_config.json and
  MSC_CONFIG is set for the MSC library.
  If MULTISTORAGECLIENT_CONFIGURATION is missing, setup_msc_config() is a best-effort sanitizer:
  it will only unset invalid MSC_CONFIG values (e.g., a directory) but will not invent a new config.
  Remote sync helpers ensure a minimal empty config exists *when they actually perform I/O*.
- Remote sync: cloud/backed paths go through MSC (s3://, msc://, gs://, ais://).
  Plain http(s) file inputs are downloaded directly unless path_mapping rewrites them to MSC.
- AWS/cloud credentials: not set by this module; MSC and boto use the standard chain
  (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, profile, IAM role, etc.).
- NVCF: progress tracking and VLM endpoint detection from env.
"""

import ast
import json
import logging
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import unquote, urlparse

import multistorageclient as msc
import multistorageclient.shortcuts as msc_shortcuts
from al_utils.media_paths import IMAGE_EXTS, VIDEO_EXTS
from config.normalize import resolve_input_path

DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0


def http_timeout_seconds() -> float:
    """Return the HTTP download timeout, overridable with ``HTTP_TIMEOUT_S`` seconds."""
    raw = os.getenv("HTTP_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_HTTP_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_HTTP_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_HTTP_TIMEOUT_SECONDS


# =============================================================================
# MSC Configuration
# =============================================================================


def setup_msc_config() -> None:
    """
    Setup MSC from MULTISTORAGECLIENT_CONFIGURATION (JSON string).

    Writes config to /tmp/msc_config.json and sets MSC_CONFIG for the library.
    Also unsets empty AWS_PROFILE / AWS_DEFAULT_PROFILE so boto credential chain
    is not confused by docker-compose pass-through (e.g. AWS_PROFILE: ${AWS_PROFILE:-}).
    """
    _load_secrets_from_file()

    # Empty profile vars make boto try to load profile ""; unset them when empty.
    for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)

    config_str = os.environ.get("MULTISTORAGECLIENT_CONFIGURATION", "").strip()
    if not config_str:
        # Be defensive: some environments (e.g. docker-compose pass-through) may set MSC_CONFIG
        # to an invalid value (like a directory). MSC will then fail with:
        #   ValueError: malformed MSC config file: <path>, exception: IsADirectoryError
        #
        # If the user did not provide MULTISTORAGECLIENT_CONFIGURATION, we should *not* force MSC
        # to read a bogus config file. Keep a valid MSC_CONFIG file path if it exists.
        raw = os.environ.get("MSC_CONFIG", "").strip()
        if raw:
            try:
                p = Path(raw)
                if (not p.exists()) or p.is_dir():
                    os.environ.pop("MSC_CONFIG", None)
            except Exception:
                # If parsing fails for any reason, unset it (best-effort).
                os.environ.pop("MSC_CONFIG", None)
        return

    try:
        _setup_msc_config(config_str)
    except Exception as e:
        print(f"Failed to setup MSC config: {e}")


def _load_secrets_from_file() -> None:
    """Load secrets from /var/secrets/secrets.json and export to env."""
    secrets_path = "/var/secrets/secrets.json"
    if not os.path.exists(secrets_path):
        return
    try:
        with open(secrets_path, "r") as f:
            secrets = json.load(f)
        if not isinstance(secrets, dict):
            return
        for key, value in secrets.items():
            if key == "MULTISTORAGECLIENT_CONFIGURATION":
                # Preserve the raw config in env so path_mapping is accessible.
                os.environ[key] = value if isinstance(value, str) else json.dumps(value)
                _setup_msc_config(value)
            else:
                os.environ[key] = str(value)
    except Exception as e:
        print(f"Failed to load secrets from {secrets_path}: {e}")


def _setup_msc_config(config: object) -> None:
    """Write MSC config to /tmp/msc_config.json and set MSC_CONFIG."""
    # Parse config if it's a JSON string
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = ast.literal_eval(config)

    if isinstance(config, dict):
        print(f"MSC config parsed. Top-level keys: {sorted(list(config.keys()))}")
        path_mapping = config.get("path_mapping", {})
        if isinstance(path_mapping, dict):
            print(f"MSC path_mapping keys: {sorted(list(path_mapping.keys()))}")

    config_path = "/tmp/msc_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    os.environ["MSC_CONFIG"] = config_path
    print(f"MSC config setup from environment variable: {config_path}")

    try:
        msc_shortcuts._reinitialize_after_fork()
    except Exception as e:
        print(f"Failed to reinitialize MSC after fork: {e}")


def _ensure_msc_config_for_io() -> None:
    """
    Ensure MSC has a usable config file path before performing remote I/O.

    Some MSC versions may attempt to read the current working directory as a config file when
    MSC_CONFIG is missing/empty, causing a crash. We avoid that by writing a minimal empty config
    only at the moment we need MSC operations (download/upload/list/delete).
    """
    # If user provided raw JSON config, prefer it (keeps path_mapping available).
    config_str = os.environ.get("MULTISTORAGECLIENT_CONFIGURATION", "").strip()
    if config_str:
        _setup_msc_config(config_str)
        return

    raw = os.environ.get("MSC_CONFIG", "").strip()
    if raw:
        try:
            p = Path(raw)
            if p.exists() and (not p.is_dir()):
                return
        except Exception:
            pass

    _setup_msc_config({})


def get_msc_path_mapping() -> dict:
    """
    Get path_mapping from MSC configuration.

    Returns:
        Dictionary mapping remote prefixes to MSC-backed prefixes.
    """
    config_str = os.environ.get("MULTISTORAGECLIENT_CONFIGURATION", "")
    if not config_str:
        # Fall back to MSC_CONFIG file if present (set by _setup_msc_config).
        config_path = os.environ.get("MSC_CONFIG", "")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_str = f.read()
            except Exception:
                return {}
        else:
            return {}

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        try:
            config = ast.literal_eval(config_str)
        except Exception:
            return {}
    except Exception:
        return {}

    try:
        return {k: v for k, v in config.get("path_mapping", {}).items() if k.endswith("/")}
    except Exception:
        return {}


def convert_to_msc_url(url: str) -> str:
    """
    Convert a URL to msc:// format using configured path mappings.

    This function uses the multistorageclient configuration's path_mapping to
    convert URLs (https://, s3://, gs://, etc.) to an MSC-backed URL.
    This approach is cloud-agnostic and supports any storage backend that MSC supports.

    Args:
        url: URL that may be mapped in MSC configuration

    Returns:
        MSC URL (msc://profile/path) if mapping found, original URL otherwise
    """
    if url.startswith("msc://"):
        return url

    try:
        path_mapping = get_msc_path_mapping()
        if not path_mapping:
            return url

        # Find the best (longest) matching prefix
        best_src = max((path for path in path_mapping if url.startswith(path)), key=len, default=None)

        if best_src:
            dest = path_mapping[best_src]
            resolved = dest + url[len(best_src) :]
            print(f"Converted URL to MSC: {url} → {resolved}")
            return resolved
        return url
    except Exception as e:
        print(f"Failed to convert URL {url} to MSC: {e}")
        return url


def normalize_remote_prefix(path: str) -> str:
    """Normalize duplicate slashes in the path portion of a remote URL or prefix."""
    s = str(path or "").strip()
    if "://" not in s:
        return s
    scheme, rest = s.split("://", 1)
    trailing = "/" if rest.endswith("/") else ""
    parts = [p for p in rest.split("/") if p]
    return f"{scheme}://{'/'.join(parts)}{trailing}"


def normalize_remote_prefix_for_compare(path: str) -> str:
    """Normalize a remote prefix for equality checks, ignoring trailing slash."""
    return normalize_remote_prefix(path).rstrip("/")


def remote_child_prefix(parent: str, child: str) -> str:
    """Return a normalized remote child prefix without using local path semantics."""
    return normalize_remote_prefix_for_compare(f"{normalize_remote_prefix_for_compare(parent)}/{child.strip('/')}")


# =============================================================================
# Remote Path Utilities
# =============================================================================


# MSC-backed remote prefixes. HTTP(S) is handled separately for direct file downloads.
MSC_REMOTE_PREFIXES = ("s3://", "msc://", "gs://", "ais://")
HTTP_PREFIXES = ("http://", "https://")
# All remote prefixes
REMOTE_PREFIXES = MSC_REMOTE_PREFIXES + HTTP_PREFIXES
# NOTE: Despite the name, the unified pipeline can also operate on a single image
# input for VLM/MCQ window stages (treated as a 1-frame clip). Keep this in sync
# with the canonical media policy so remote staging does not admit unsupported
# containers.
VIDEO_EXTENSIONS = tuple(sorted(VIDEO_EXTS | IMAGE_EXTS))


def is_http_path(path: str) -> bool:
    """Check if path is an HTTP(S) URL."""
    return any(path.startswith(prefix) for prefix in HTTP_PREFIXES)


def is_remote_path(path: str) -> bool:
    """
    Check if a path is remote (s3://, msc://, gs://, etc.).
    """
    return any(path.startswith(prefix) for prefix in REMOTE_PREFIXES)


def _remote_basename(remote_path: str) -> str:
    """Return a filesystem-friendly basename for a remote URL/path."""
    path_without_query = remote_path.rstrip("/").split("?")[0]
    return os.path.basename(unquote(path_without_query))


def _download_http_file(url: str, local_file: str, *, timeout: Optional[float] = None) -> None:
    """Download a single public HTTP(S) file without requiring MSC configuration."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"HTTP download only supports http:// or https:// URLs, got {scheme or '<none>'!r}")
    parent = os.path.dirname(local_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=timeout if timeout is not None else http_timeout_seconds()) as response,
        open(local_file, "wb") as out,
    ):
        shutil.copyfileobj(response, out)


def sync_remote_to_local(
    remote_path: str,
    local_dir: str,
    extensions: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> str:
    """
    Sync files from remote path to local directory.

    Unmapped HTTP(S) paths support direct single-file downloads only. Directory
    sync and uploads require an MSC-backed scheme or path_mapping rewrite.

    Args:
        remote_path: Remote path (s3://, msc://, gs://, http://, https://, etc.)
        local_dir: Local directory to sync files to.
        extensions: Optional tuple of extensions to filter (e.g., VIDEO_EXTENSIONS).
        verbose: Print progress messages.
    """
    os.makedirs(local_dir, exist_ok=True)
    _ensure_msc_config_for_io()

    # Convert URL to an MSC-backed path using path_mapping if available.
    converted = convert_to_msc_url(remote_path)
    if converted != remote_path:
        remote_path = converted

    if verbose:
        print(f"Syncing remote path to local: {remote_path} -> {local_dir}")

    try:
        # If remote_path points to a single file, download it directly.
        # Strip query string so https://host/file.mp4?token=x is treated as file.mp4
        filename = _remote_basename(remote_path)
        if filename:
            is_video = filename.lower().endswith(VIDEO_EXTENSIONS)
            if (extensions and any(filename.lower().endswith(ext) for ext in extensions)) or (
                not extensions and is_video
            ):
                local_file = os.path.join(local_dir, filename)
                if verbose:
                    print(f"Downloading file: {filename}")
                if is_http_path(remote_path):
                    _download_http_file(remote_path, local_file)
                else:
                    msc.download_file(remote_path, local_file)
                if verbose:
                    print(f"Synced 1 file to {local_dir}")
                return local_dir

        if is_http_path(remote_path):
            raise RuntimeError(
                f"Direct HTTP(S) remote input must point to a single supported file: {remote_path}. "
                "Use MULTISTORAGECLIENT_CONFIGURATION.path_mapping for HTTP(S) storage prefixes."
            )

        # Directory or non-matching path: list and download each file
        remote_path_normalized = remote_path.rstrip("/") + "/"
        obj_list = list(msc.list(url=remote_path_normalized, include_directories=False))

        files_synced = 0
        for obj_metadata in obj_list:
            remote_file = obj_metadata.key
            # Skip directory entries (MSC may still list them when include_directories=False on some backends)
            if getattr(obj_metadata, "type", None) == "directory":
                if verbose:
                    print(f"Skipping directory entry: {remote_file}")
                continue
            # Key can be bucket/path or msc://profile/bucket/path; get last segment
            key_stripped = remote_file.rstrip("/")
            filename = os.path.basename(key_stripped)

            # Skip directory placeholders (S3-compatible backends may use key ending in /. or key ".")
            if not filename or filename in (".", ".."):
                if verbose:
                    print(f"Skipping placeholder (empty or . / ..): {remote_file}")
                continue
            if key_stripped.endswith("/.") or key_stripped == ".":
                if verbose:
                    print(f"Skipping directory placeholder: {remote_file}")
                continue

            # Filter by extension if specified
            if extensions and (not any(filename.lower().endswith(ext) for ext in extensions)):
                continue

            # Preserve relative path under remote prefix. Keys from msc.list are msc://profile/path (path may
            # omit bucket); remote is s3://bucket/path. Compare path-after-profile to path-after-bucket.
            # Normalize remote prefix + keys to avoid subtle '//' mismatches.
            key_after_protocol = remote_file.split("://", 1)[-1].lstrip("/")
            key_parts = [p for p in key_after_protocol.split("/") if p != ""]
            key_path_under_profile = "/".join(key_parts[1:]) if len(key_parts) > 1 else ""
            key_path_under_profile = key_path_under_profile.strip("/")

            remote_after_protocol = remote_path_normalized.split("://", 1)[-1].lstrip("/")
            remote_parts = [p for p in remote_after_protocol.split("/") if p != ""]
            remote_path_under_bucket = "/".join(remote_parts[1:]) if len(remote_parts) > 1 else ""
            remote_path_under_bucket = remote_path_under_bucket.strip("/")
            remote_prefix = (remote_path_under_bucket + "/") if remote_path_under_bucket else ""
            if key_path_under_profile.startswith(remote_prefix) or key_path_under_profile == remote_path_under_bucket:
                rel_path = (
                    key_path_under_profile[len(remote_prefix) :].lstrip("/")
                    if key_path_under_profile.startswith(remote_prefix)
                    else ""
                )
            else:
                rel_path = filename
            if not rel_path:
                continue
            local_file = os.path.join(local_dir, rel_path.replace("/", os.sep))
            parent = os.path.dirname(local_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if verbose:
                print(f"Downloading: {rel_path}")
            msc.download_file(remote_file, local_file)
            files_synced += 1

        if verbose:
            print(f"Synced {files_synced} files to {local_dir}")
        if files_synced == 0 and extensions:
            raise RuntimeError(f"No files with extensions {extensions} found under {remote_path_normalized}")
        return local_dir

    except Exception as e:
        print(f"Error syncing remote path: {e}")
        raise


def sync_local_to_remote(
    local_dir: str,
    remote_path: str,
    verbose: bool = True,
) -> None:
    """
    Sync files from local directory to remote path.

    Args:
        local_dir: Local directory containing files to upload.
        remote_path: Remote path to upload files to (s3://, msc://, gs://, https://, etc.)
        verbose: Print progress messages.
    """
    # Convert URL to msc:// using path_mapping
    _ensure_msc_config_for_io()
    converted = convert_to_msc_url(remote_path)
    if converted != remote_path:
        remote_path = converted
    elif is_http_path(remote_path):
        raise RuntimeError(
            f"Cannot upload to HTTP(S) URL without path_mapping: {remote_path}. "
            "Configure path_mapping in MULTISTORAGECLIENT_CONFIGURATION."
        )

    if verbose:
        print(f"Syncing local to remote: {local_dir} -> {remote_path}")

    try:
        remote_path_normalized = remote_path.rstrip("/") + "/"
        files_uploaded = 0

        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_dir)
                remote_file = remote_path_normalized + rel_path.replace(os.sep, "/")

                if verbose:
                    print(f"Uploading: {rel_path}")
                msc.upload_file(remote_file, local_file)
                files_uploaded += 1

        if verbose:
            print(f"Uploaded {files_uploaded} files to {remote_path}")

    except Exception as e:
        print(f"Error syncing to remote path: {e}")
        raise


# =============================================================================
# NVCF Environment / Progress Tracking
# =============================================================================


def is_nvcf_task_env() -> bool:
    """Return true when running inside an NVCF task environment."""
    return bool(os.getenv("NVCT_TASK_ID") and os.getenv("NVCT_PROGRESS_FILE_PATH"))


class NVCFProgressTracker:
    """
    Simple NVCF progress tracker for task environments.

    Detects NVCF task environment variables and writes progress updates
    to the progress file path specified by NVCT_PROGRESS_FILE_PATH.
    """

    def __init__(self):
        """Initialize NVCF progress tracker."""
        self.is_nvcf = is_nvcf_task_env()

        if self.is_nvcf:
            self.task_id = os.getenv("NVCT_TASK_ID")
            self.progress_path = os.getenv("NVCT_PROGRESS_FILE_PATH")
            print("NVCF Task environment detected")
            # Create the progress file immediately so NVCF marks the container ready
            self.update(1, "Task started")
        else:
            self.task_id = None
            self.progress_path = None

    def update(self, percent: float, message: str = "Processing") -> None:
        """
        Update progress if in NVCF environment.

        Args:
            percent: Progress percentage (0-100).
            message: Progress message to display.
        """
        if not self.is_nvcf:
            return

        try:
            payload = {
                "taskId": self.task_id,
                "percentComplete": int(percent),
                "message": message,
                "lastUpdatedAt": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.progress_path, "w") as f:
                json.dump(payload, f, indent=2)

        except Exception as e:
            print(f"Failed to update NVCF progress: {e}")


# =============================================================================
# NVCF VLM Endpoint Detection
# =============================================================================


def detect_nvcf_vlm_endpoint() -> Tuple[Optional[str], Optional[str]]:
    """
    Detect VLM endpoint from environment variables.

    Checks for VLM_BASE_URL and VLM_MODEL environment variables.
    """
    vlm_base_url = os.environ.get("VLM_BASE_URL")
    vlm_model = os.environ.get("VLM_MODEL")

    if vlm_base_url:
        print(f"VLM endpoint detected from environment: {vlm_base_url}")
        if vlm_model:
            print(f"VLM model: {vlm_model}")
        return vlm_base_url, vlm_model

    return None, None


def detect_nvcf_llm_endpoint() -> Tuple[Optional[str], Optional[str]]:
    """
    Detect LLM endpoint from environment variables.

    Checks for LLM_BASE_URL and LLM_MODEL environment variables.
    """
    llm_base_url = os.environ.get("LLM_BASE_URL")
    llm_model = os.environ.get("LLM_MODEL")

    if llm_base_url:
        print(f"LLM endpoint detected from environment: {llm_base_url}")
        if llm_model:
            print(f"LLM model: {llm_model}")
        return llm_base_url, llm_model

    return None, None


# =============================================================================
# Local/Remote I/O Materialization Helpers
# =============================================================================


def localize_path_to_dir(
    path: Optional[str],
    *,
    dst_dir: Path,
    logger: Any,
    extensions: Tuple[str, ...],
    config_dir: Path,
    repo_root: Path,
) -> Optional[Path]:
    """
    Make sure the given path is available as a local filesystem file.
    - remote: download into dst_dir and return the downloaded file path
      (direct HTTP(S) single-file download, or MSC-backed paths)
    - local: return the resolved local path directly (no copying/symlinking)
    Returns the local file path (or None if input path is None/empty).
    """
    if path is None:
        return None
    v = str(path or "").strip()
    if not v:
        return None

    if is_remote_path(v):
        dst_dir.mkdir(parents=True, exist_ok=True)
        name = _remote_basename(v) or "file"
        out = dst_dir / name
        sync_remote_to_local(remote_path=v, local_dir=str(dst_dir), extensions=extensions, verbose=False)
        if out.exists():
            return out
        # Strict: require the downloaded filename to match the remote basename.
        allowed_exts = {str(e).lower() for e in (extensions or ()) if str(e).strip()}
        cands = [p for p in dst_dir.iterdir() if p.is_file() and (not allowed_exts or p.suffix.lower() in allowed_exts)]
        preview = ", ".join(sorted(p.name for p in cands)[:10])
        more = "" if len(cands) <= 10 else f" (+{len(cands) - 10} more)"
        raise RuntimeError(
            f"Failed to download remote file (expected filename {name!r} in {dst_dir}, got: {preview}{more}): {v}"
        )

    # Local input path resolution:
    # - Prefer repo_root-relative for plain relative paths.
    # - Fall back to config_dir when the repo_root candidate does not exist.
    src = Path(v).expanduser()
    if not src.is_absolute():
        try:
            src = Path(resolve_input_path(v, config_dir=config_dir, repo_root=repo_root)).expanduser()
        except Exception:
            src = Path(v).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Local input not found: {src}")
    return src.resolve()


def upload_file_to_remote(*, local_src: Path, remote_dst: str) -> None:
    """Upload a single local file to a remote path via MSC."""
    dst = str(remote_dst or "").strip()
    dst2 = convert_to_msc_url(dst)
    if dst2 == dst and is_http_path(dst2):
        raise RuntimeError(
            f"Cannot upload to HTTP(S) URL without path_mapping: {dst2}. "
            "Configure MULTISTORAGECLIENT_CONFIGURATION.path_mapping."
        )
    msc.upload_file(dst2, str(local_src))


def materialize_move(*, src: Path, dst: str, dry_run: bool, logger: logging.Logger) -> None:
    """
    Materialize an artifact to the requested output path.
    - local path: move (rename) into place (no duplication)
    - remote path: upload (copy)
    """
    if not src.exists():
        raise FileNotFoundError(f"Artifact not found: {src}")
    d = str(dst or "").strip()
    if not d:
        raise ValueError("Empty output path")

    if dry_run:
        logger.info(f"[materialize] {src} -> {d}")
        return

    if is_remote_path(d):
        upload_file_to_remote(local_src=src, remote_dst=d)
        logger.info(f"[materialize] uploaded: {src} -> {d}")
        return

    dp = Path(d).expanduser()
    dp.parent.mkdir(parents=True, exist_ok=True)
    if dp.exists():
        dp.unlink()
    shutil.move(str(src), str(dp))
    logger.info(f"[materialize] moved: {src} -> {dp}")
