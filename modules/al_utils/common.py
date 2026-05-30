# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import logging
import os
import runpy
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


def get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Small helper: nested dict get via dot-path (e.g. 'tracking.scene_id')."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class StageResult:
    name: str
    rc: int
    started_at: float
    ended_at: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


def ensure_dir(p: Path) -> None:
    # Global dry-run switch used to guarantee no filesystem writes, even if
    # stage planning code still calls ensure_dir().
    if str(os.getenv("AUTO_LABELING_DRY_RUN", "")).strip().lower() in {"1", "true", "yes", "y"}:
        return
    p.mkdir(parents=True, exist_ok=True)


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
                st.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:  # type: ignore[override]
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


@contextlib.contextmanager
def _pushd(new_dir: Path):
    old = Path.cwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _temp_environ(env: Dict[str, str]):
    old = dict(os.environ)
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def run_py_file(
    *,
    name: str,
    py_file: Path,
    argv: List[str],
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    log_dir: Optional[Path] = None,
    pipeline_log: Optional[Path] = None,
    logger=None,
) -> StageResult:
    started = time.time()

    if not py_file.exists():
        raise FileNotFoundError(f"Stage '{name}' script not found: {py_file}")

    stage_log_fp: Optional[io.TextIOWrapper] = None
    pipe_log_fp: Optional[io.TextIOWrapper] = None
    rc = 0
    try:
        if log_dir is not None:
            ensure_dir(log_dir)
            stage_log_fp = (log_dir / f"{name}.log").open("a", encoding="utf-8")
        if pipeline_log is not None:
            ensure_dir(pipeline_log.parent)
            pipe_log_fp = pipeline_log.open("a", encoding="utf-8")

        header = (
            "\n"
            + "=" * 80
            + f"\n[{name}] START {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + f"[{name}] CWD: {str(cwd or Path.cwd())}\n"
            + f"[{name}] CMD: python {py_file} "
            + " ".join(argv)
            + "\n"
            + "=" * 80
            + "\n"
        )
        for fp in [stage_log_fp, pipe_log_fp]:
            if fp is not None:
                fp.write(header)
                fp.flush()
        sys.stdout.write(header)

        tee_out = _Tee(*(s for s in [sys.stdout, stage_log_fp, pipe_log_fp] if s is not None))
        tee_err = _Tee(*(s for s in [sys.stderr, stage_log_fp, pipe_log_fp] if s is not None))

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            with _temp_environ(extra_env or {}):
                with _pushd(cwd or Path.cwd()):
                    old_argv = sys.argv
                    sys.argv = [str(py_file)] + argv
                    old_sys_path = list(sys.path)
                    try:
                        # When executing a Python file via runpy.run_path, Python does NOT reliably
                        # behave like "python /path/to/script.py" with respect to import roots.
                        # Ensure the script's directory is on sys.path so vendored modules
                        # (e.g. modules/detection_and_tracking/boosttrack) are importable.
                        script_dir = str(py_file.parent)
                        if script_dir not in sys.path:
                            sys.path.insert(0, script_dir)
                        runpy.run_path(str(py_file), run_name="__main__")
                    except SystemExit as e:
                        rc = int(e.code) if isinstance(e.code, int) else 1
                        if rc != 0:
                            raise
                    except Exception:
                        rc = 1
                        raise
                    finally:
                        sys.argv = old_argv
                        sys.path[:] = old_sys_path
    finally:
        ended = time.time()
        footer = f"[{name}] END   {time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc} ({ended - started:.2f}s)\n"
        for fp in [stage_log_fp, pipe_log_fp]:
            if fp is not None:
                fp.write(footer)
                fp.flush()
                fp.close()
        sys.stdout.write(footer)

    return StageResult(name=name, rc=rc, started_at=started, ended_at=ended)


@contextmanager
def stage_log_file(name: str, log_dir: Optional[Path]) -> Generator[None, None, None]:
    """Context manager that tees all root-logger output to log_dir/<name>.log for the duration.

    Writes a START/END header matching the run_cmd format so all stage logs look consistent.
    No-op when log_dir is None.
    """
    if log_dir is None:
        yield
        return

    ensure_dir(log_dir)
    handler = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'=' * 80}\n[{name}] START {started}\n{'=' * 80}\n"
    handler.stream.write(header)
    handler.stream.flush()
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        footer = f"[{name}] END   {ended} ({elapsed:.2f}s)\n"
        handler.stream.write(footer)
        handler.stream.flush()
        root.removeHandler(handler)
        handler.close()


def resolve_gpu_list(gpu_ids: str | int | None) -> list[int]:
    """Parse pipeline.gpu_ids into a list of non-negative GPU indices.

    - None / "" / "all"  → all visible CUDA devices; fallback [0] if unavailable
    - int 0              → [0]  (fixes falsy ``0 or 'all'`` bug)
    - "2,3"              → [2, 3]
    - " 2 , 3 "          → [2, 3]
    - non-numeric token  → raises ValueError
    - negative value     → raises ValueError
    - empty after parse  → [0]
    """
    if gpu_ids is None:
        return _all_gpu_ids()

    if isinstance(gpu_ids, int):
        if gpu_ids < 0:
            raise ValueError(f"GPU ID must be non-negative, got {gpu_ids}")
        return [gpu_ids]

    s = str(gpu_ids).strip()
    if s.lower() in {"", "all"}:
        return _all_gpu_ids()

    parts = [p.strip() for p in s.split(",")]
    result: list[int] = []
    for part in parts:
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            raise ValueError(f"Invalid GPU ID {part!r} in gpu_ids={gpu_ids!r}") from None
        if val < 0:
            raise ValueError(f"GPU ID must be non-negative, got {val} in gpu_ids={gpu_ids!r}")
        result.append(val)
    return result if result else [0]


def _all_gpu_ids() -> list[int]:
    try:
        import torch

        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


def run_cmd(
    *,
    name: str,
    cmd: List[str],
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    log_dir: Optional[Path] = None,
    pipeline_log: Optional[Path] = None,
    logger=None,
) -> StageResult:
    """
    Run an external command (subprocess) with stdout/stderr tee'd to:
    - console
    - per-stage log file (log_dir/<name>.log)
    - optional pipeline_log
    """
    started = time.time()

    stage_log_fp: Optional[io.TextIOWrapper] = None
    pipe_log_fp: Optional[io.TextIOWrapper] = None
    rc = 0
    try:
        if log_dir is not None:
            ensure_dir(log_dir)
            stage_log_fp = (log_dir / f"{name}.log").open("a", encoding="utf-8")
        if pipeline_log is not None:
            ensure_dir(pipeline_log.parent)
            pipe_log_fp = pipeline_log.open("a", encoding="utf-8")

        header = (
            "\n"
            + "=" * 80
            + f"\n[{name}] START {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + f"[{name}] CWD: {str(cwd or Path.cwd())}\n"
            + f"[{name}] CMD: {shlex.join(cmd)}\n"
            + "=" * 80
            + "\n"
        )
        for fp in [stage_log_fp, pipe_log_fp]:
            if fp is not None:
                fp.write(header)
                fp.flush()
        sys.stdout.write(header)

        tee_out = _Tee(*(s for s in [sys.stdout, stage_log_fp, pipe_log_fp] if s is not None))

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tee_out.write(line)
        rc = int(proc.wait())
        if rc != 0:
            raise SystemExit(rc)
    finally:
        ended = time.time()
        footer = f"[{name}] END   {time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc} ({ended - started:.2f}s)\n"
        for fp in [stage_log_fp, pipe_log_fp]:
            if fp is not None:
                fp.write(footer)
                fp.flush()
                fp.close()
        sys.stdout.write(footer)

    return StageResult(name=name, rc=rc, started_at=started, ended_at=ended)


__all__ = [
    "StageResult",
    "ensure_dir",
    "get",
    "resolve_gpu_list",
    "run_cmd",
    "run_py_file",
    "stage_log_file",
]
