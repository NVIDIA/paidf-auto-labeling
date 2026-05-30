# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
from filelock import SoftFileLock
from huggingface_hub import hf_hub_download


def resolve_ckpts_root(*, repo_root: Path, model_cache_path: Optional[str] = None) -> Path:
    """
    Resolve the checkpoints root directory (shared across all models).

    Precedence:
      1) env MODEL_CACHE_PATH
      2) config.pipeline.model_cache_path
      3) /workspace/ckpts (Docker default)
      4) <repo_root>/ckpts (local default)
    """
    env_cache = str(os.getenv("MODEL_CACHE_PATH", "")).strip()
    if env_cache.lower() == "none":
        env_cache = ""
    cfg_cache = str(model_cache_path or "").strip()
    if cfg_cache.lower() == "none":
        cfg_cache = ""

    if env_cache:
        return Path(env_cache).expanduser().resolve()
    if cfg_cache:
        return Path(cfg_cache).expanduser().resolve()

    docker_default = Path("/workspace/ckpts")
    if docker_default.exists():
        return docker_default.resolve()

    return (Path(repo_root) / "ckpts").resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_url_downloaded(
    *,
    url: str,
    dst: Path,
    timeout_s: int = 600,
    sha256: Optional[str] = None,
    min_bytes: int = 1024 * 1024,
) -> None:
    """
    Download url -> dst if missing.

    - Uses a lock file to avoid concurrent downloads.
    - Writes to *.partial then renames (atomic) to avoid corrupt partials.
    - Optional SHA256 verification.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock = SoftFileLock(str(dst) + ".lock", timeout=timeout_s)
    with lock:
        if dst.exists() and dst.stat().st_size > 0:
            if sha256:
                got = _sha256_file(dst)
                exp = str(sha256).strip().lower()
                if got.lower() != exp:
                    raise RuntimeError(f"SHA256 mismatch for {dst}: expected {exp}, got {got}")
            return

        tmp = dst.with_suffix(dst.suffix + ".partial")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

        r = requests.get(str(url), stream=True, timeout=timeout_s)
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)

        size = tmp.stat().st_size if tmp.exists() else 0
        if size < int(min_bytes):
            try:
                tmp.unlink()
            except Exception:
                pass
            raise RuntimeError(f"Downloaded file too small ({size} bytes): {dst} from {url}")

        tmp.replace(dst)

        if sha256:
            got = _sha256_file(dst)
            exp = str(sha256).strip().lower()
            if got.lower() != exp:
                raise RuntimeError(f"SHA256 mismatch for {dst}: expected {exp}, got {got}")


def ensure_hf_file(
    *,
    repo_id: str,
    filename: str,
    dst: Path,
    hf_token: Optional[str] = None,
    timeout_s: int = 600,
) -> None:
    """
    Download a HuggingFace file into dst if missing.

    Uses a per-destination lock for concurrency safety. Writes into dst.parent and then ensures
    the final file is exactly at dst.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock = SoftFileLock(str(dst) + ".lock", timeout=timeout_s)
    with lock:
        if dst.exists() and dst.stat().st_size > 0:
            return
        path = hf_hub_download(
            repo_id=str(repo_id),
            filename=str(filename),
            token=hf_token if hf_token else None,
            local_dir=str(dst.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        p = Path(str(path))
        if p.resolve() != dst.resolve():
            # Ensure the file ends up exactly at dst (no symlink surprises).
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
            p.replace(dst)
