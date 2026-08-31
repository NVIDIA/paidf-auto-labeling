# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SeedVR2 super-resolution resolver.

The resolver owns runtime preparation and subprocess execution for SeedVR2.
It intentionally exposes a narrow in-process API to the task layer while
keeping model-specific concerns (checkpoints, GPU routing, torchrun flags)
inside this backend.
"""

from __future__ import annotations

import errno
import importlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from core import ScenePaths
from core.media.video_codecs import prepare_video_decode

from super_resolution.config import SeedVR2Config
from super_resolution.media_formats import is_supported_sr_image_path, sr_output_name_for_input
from super_resolution.resolver import Resolver, SrResult

_SEEDVR_REQUIRED_PATHS: tuple[str, ...] = (
    "common",
    "configs_3b",
    "configs_7b",
    "data",
    "models",
    "pos_emb.pt",
    "neg_emb.pt",
    "projects",
)

_VARIANT_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "seedvr2_3b": ("ema_vae.pth", "seedvr2_ema_3b.pth"),
    "seedvr2_7b": ("ema_vae.pth", "seedvr2_ema_7b.pth"),
}

_NO_CUDA_DEVICES_MESSAGE = (
    "No CUDA devices were detected while resolving gpu_ids='all'. Run the container with "
    "GPUs enabled, set gpu_ids to explicit visible CUDA device ids, or use a CPU-capable "
    "resolver."
)


@dataclass(frozen=True)
class CommandInvocation:
    """External command invocation requested by a resolver."""

    name: str
    command: list[str]
    cwd: Path
    env: Mapping[str, str]
    log_dir: Path | None
    timeout_s: float | None


class CommandRunner(Protocol):
    """Callable used to execute a command invocation."""

    def __call__(self, invocation: CommandInvocation) -> None:
        """Execute ``invocation`` or raise ``CommandRunnerError`` on runner failure."""


class CommandRunnerError(RuntimeError):
    """Raised by custom command runners when command execution fails."""


class SubprocessCommandRunner:
    """Run resolver commands with ``subprocess.run``."""

    def __call__(self, invocation: CommandInvocation) -> None:
        env = os.environ.copy()
        env.update(invocation.env)
        stdout = None
        log_handle = None
        try:
            if invocation.log_dir is not None:
                invocation.log_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (invocation.log_dir / f"{invocation.name}.log").open(
                    "a",
                    encoding="utf-8",
                )
                log_handle.write(f"CMD: {' '.join(invocation.command)}\n")
                log_handle.flush()
                stdout = log_handle
            subprocess.run(
                invocation.command,
                cwd=invocation.cwd,
                env=env,
                check=True,
                stdout=stdout,
                stderr=subprocess.STDOUT if stdout is not None else None,
                timeout=invocation.timeout_s,
            )
        finally:
            if log_handle is not None:
                log_handle.close()


def resolve_gpu_ids(gpu_ids: str | int | None) -> list[int]:
    """Parse a GPU id setting into non-negative physical CUDA ids."""
    if gpu_ids is None:
        return _all_gpu_ids()
    if isinstance(gpu_ids, int):
        if gpu_ids < 0:
            raise ValueError(f"GPU id must be non-negative, got {gpu_ids}")
        return [gpu_ids]

    raw = str(gpu_ids).strip()
    if raw.lower() in {"", "all"}:
        return _all_gpu_ids()

    out: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError:
            raise ValueError(f"Invalid GPU id {token!r} in gpu_ids={gpu_ids!r}") from None
        if gpu_id < 0:
            raise ValueError(f"GPU id must be non-negative, got {gpu_id}")
        out.append(gpu_id)
    return out or _all_gpu_ids()


def resolve_model_cache_root(model_cache_path: str | None) -> Path:
    """Resolve the shared model cache root."""
    env_cache = _clean_optional_path(os.getenv("MODEL_CACHE_PATH"))
    cfg_cache = _clean_optional_path(model_cache_path)
    if env_cache is not None:
        return env_cache
    if cfg_cache is not None:
        return cfg_cache
    container_default = Path("/models")
    if container_default.exists():
        return container_default.resolve()
    return (Path.cwd() / "ckpts").resolve()


def resolve_seedvr_source_root(seedvr_root: str | None) -> Path:
    """Resolve the SeedVR source root used by the torchrun module."""
    cfg_root = _clean_optional_path(seedvr_root)
    env_root = _clean_optional_path(os.getenv("SEEDVR_ROOT"))
    candidates = [p for p in (cfg_root, env_root, Path("/opt/seedvr")) if p is not None]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(p) for p in candidates) or "<none>"
    raise RuntimeError(
        "SeedVR2 source root was not found. Set SeedVR2Config.seedvr_root or "
        f"SEEDVR_ROOT to a directory containing SeedVR projects. Checked: {rendered}"
    )


def validate_seedvr_source_root(*, source_root: Path) -> Path:
    """Validate and return a SeedVR source root."""
    source_root = source_root.expanduser().resolve()
    for name in _SEEDVR_REQUIRED_PATHS:
        target = source_root / name
        if not target.exists():
            raise RuntimeError(f"SeedVR source root is missing required path: {target}")
    return source_root


def prepare_seedvr_runtime_root(
    *,
    source_root: Path,
    runtime_root: Path,
    ckpts_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Create the writable SeedVR runtime root expected by the legacy runner.

    SeedVR reads configs, prompt embeddings, source modules, and ``./ckpts`` relative to its
    current working directory. This mirrors ``/home/metrosdg/pseudo-labeling``:
    the runtime root lives in the shared model cache, links source assets from
    ``SEEDVR_ROOT``, and links ``ckpts`` to ``<model-cache>/seedvr2``.
    """
    source_root = validate_seedvr_source_root(source_root=source_root)
    ckpts_dir = Path(ckpts_dir).expanduser().resolve()
    if not ckpts_dir.is_dir():
        raise RuntimeError(f"SeedVR checkpoint directory is missing: {ckpts_dir}")

    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    expected_links = set(_SEEDVR_REQUIRED_PATHS) | {"ckpts"}
    for stale in runtime_root.iterdir():
        if stale.name in expected_links:
            continue
        try:
            if stale.is_symlink() or stale.is_file():
                stale.unlink()
            elif stale.is_dir():
                shutil.rmtree(stale)
            elif stale.exists():
                raise RuntimeError(f"Unsupported stale SeedVR runtime entry: {stale}")
            else:
                continue
        except OSError as exc:
            if _is_missing_path_error(exc):
                continue
            logger.exception("Failed to remove stale SeedVR runtime entry: %s", stale)
            raise RuntimeError(f"Failed to remove stale SeedVR runtime entry: {stale}") from exc
    for name in _SEEDVR_REQUIRED_PATHS:
        _ensure_runtime_symlink(
            link_path=runtime_root / name,
            target=source_root / name,
            description=f"SeedVR {name}",
        )
    _ensure_runtime_symlink(
        link_path=runtime_root / "ckpts",
        target=ckpts_dir,
        description="SeedVR checkpoints",
    )
    logger.info(
        "SeedVR runtime root: %s (source=%s, ckpts=%s)",
        runtime_root,
        source_root,
        ckpts_dir,
    )
    return runtime_root.resolve()


class SeedVR2Resolver(Resolver):
    """SeedVR2 implementation backed by a ``torchrun`` subprocess."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        config: SeedVR2Config,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.logger = logger
        self.config = config
        if command_runner is None:
            validate_seedvr2_runtime_command()
        self._command_runner = command_runner or SubprocessCommandRunner()
        self._cache_root = resolve_model_cache_root(config.model_cache_path)
        self._ckpts_dir = self._cache_root / "seedvr2"
        if config.allow_checkpoint_download:
            ensure_seedvr2_checkpoints(
                ckpts_root=self._cache_root,
                variant=config.variant,
                allow_download=True,
                logger=logger,
            )
        else:
            validate_seedvr2_checkpoints(ckpts_root=self._cache_root, variant=config.variant)

        self._source_root = validate_seedvr_source_root(
            source_root=resolve_seedvr_source_root(config.seedvr_root)
        )
        self._seedvr_root = prepare_seedvr_runtime_root(
            source_root=self._source_root,
            runtime_root=self._cache_root / "seedvr_runtime",
            ckpts_dir=self._ckpts_dir,
            logger=self.logger,
        )
        self.logger.info(
            "SeedVR source root: %s (runtime=%s, ckpts=%s)",
            self._source_root,
            self._seedvr_root,
            self._ckpts_dir,
        )
        self._module = "super_resolution.seedvr2_torchrun"

    def run(self, media_path: Path, scene_paths: ScenePaths) -> SrResult:
        media_path = Path(media_path).expanduser().resolve()
        is_image_input = is_supported_sr_image_path(media_path)
        decoder_name: str | None = None
        if not is_image_input:
            decoder_name = prepare_video_decode(media_path).decoder_name

        gpu_list = resolve_gpu_ids(self.config.gpu_ids)
        active_gpu_list = gpu_list if self.config.use_multi_gpu else [gpu_list[0]]
        sp_size = len(active_gpu_list)
        sidecars_dir = Path(scene_paths.sidecars_dir).expanduser().resolve()
        output_path = sidecars_dir / sr_output_name_for_input(media_path)
        sidecars_dir.mkdir(parents=True, exist_ok=True)
        _remove_existing_output(output_path)

        work_dir = sidecars_dir / "_work" / "sr"
        tmp_dir = work_dir / "_tmp_window_segments"
        output_dir = work_dir / "out"
        runtime_cache_dir = work_dir / "_runtime_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_cache_dir.mkdir(parents=True, exist_ok=True)

        run_sp_size = 1 if is_image_input else sp_size
        visible_gpus = active_gpu_list[:run_sp_size]
        env = self._build_runtime_env(
            visible_gpus=visible_gpus,
            runtime_cache_dir=runtime_cache_dir,
            seedvr_root=self._seedvr_root,
        )

        invocation = CommandInvocation(
            name="sr",
            command=self._build_command(
                input_path=media_path,
                output_path=output_path,
                output_dir=output_dir,
                tmp_dir=tmp_dir,
                sp_size=run_sp_size,
                decoder_name=decoder_name,
            ),
            cwd=self._seedvr_root,
            env=env,
            log_dir=sidecars_dir / "logs",
            timeout_s=self.config.command_timeout_s,
        )

        failure_result: SrResult | None = None
        try:
            try:
                self._command_runner(invocation)
            except (
                CommandRunnerError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                OSError,
            ) as exc:
                failure_result = self._handle_failure(f"SeedVR2 subprocess failed: {exc}", exc)
            else:
                if not _is_nonempty_file(output_path):
                    failure_result = self._handle_failure(
                        f"SeedVR2 output was not created: {output_path}",
                        None,
                    )
        finally:
            if not self.config.keep_intermediates:
                shutil.rmtree(work_dir, ignore_errors=True)

        if failure_result is not None:
            return failure_result
        return SrResult(success=True, output_path=output_path)

    def _build_command(
        self,
        *,
        input_path: Path,
        output_path: Path,
        output_dir: Path,
        tmp_dir: Path,
        sp_size: int,
        decoder_name: str | None,
    ) -> list[str]:
        command = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={sp_size}",
            "--module",
            self._module,
            "--variant",
            self.config.variant,
            "--video_path",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--output_path",
            str(output_path),
            "--tmp_dir",
            str(tmp_dir),
            "--seed",
            str(self.config.seed),
            "--res_h",
            str(self.config.res_h),
            "--res_w",
            str(self.config.res_w),
            "--sp_size",
            str(sp_size),
            "--window_frames",
            str(self.config.window_frames),
            "--overlap_frames",
            str(self.config.overlap_frames),
        ]
        if self.config.out_fps is not None:
            command.extend(["--out_fps", str(self.config.out_fps)])
        if decoder_name is not None:
            command.extend(["--video_decoder", decoder_name])
        return command

    def _build_runtime_env(
        self,
        *,
        visible_gpus: list[int],
        runtime_cache_dir: Path,
        seedvr_root: Path,
    ) -> dict[str, str]:
        xdg_cache_dir = runtime_cache_dir / "xdg"
        env_paths = {
            "HOME": runtime_cache_dir / "home",
            "XDG_CACHE_HOME": xdg_cache_dir,
            "MPLCONFIGDIR": xdg_cache_dir / "matplotlib",
            "TORCH_HOME": xdg_cache_dir / "torch",
            "TORCHINDUCTOR_CACHE_DIR": xdg_cache_dir / "torchinductor",
        }
        for path in env_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        username = os.getenv("USER") or os.getenv("LOGNAME") or "sr"
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in visible_gpus),
            "SEEDVR_ROOT": str(seedvr_root),
            "SEEDVR_CKPT_DIR": str(self._ckpts_dir),
            "PYTHONPATH": _runtime_pythonpath(seedvr_root),
            "USER": username,
            "LOGNAME": username,
            **{key: str(path) for key, path in env_paths.items()},
        }

    def _handle_failure(self, message: str, exc: BaseException | None) -> SrResult:
        if self.config.empty_output_policy == "fail":
            raise RuntimeError(message) from exc
        self.logger.warning("%s (continuing without enhanced media)", message)
        return SrResult(success=False, message=message)


def validate_seedvr2_checkpoints(*, ckpts_root: Path, variant: str) -> None:
    """Raise if required SeedVR2 checkpoint files are missing."""
    missing = _missing_checkpoints(ckpts_root=ckpts_root, variant=variant)
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "SeedVR2 checkpoints are missing. Mount or prefetch the files, "
            "or set allow_checkpoint_download=true. Missing: "
            f"{rendered}"
        )


def validate_seedvr2_runtime_command() -> None:
    """Raise if the active runtime image cannot launch SeedVR2."""
    if shutil.which("torchrun") is not None:
        return
    raise RuntimeError(
        "SeedVR2 requires torchrun in PATH. Install PyTorch with a CUDA runtime "
        "compatible with the image, or use the super-resolution service image."
    )


def ensure_seedvr2_checkpoints(
    *,
    ckpts_root: Path,
    variant: str,
    allow_download: bool,
    logger: logging.Logger,
) -> None:
    """Ensure required checkpoints exist, optionally downloading from Hugging Face."""
    missing = _missing_checkpoints(ckpts_root=ckpts_root, variant=variant)
    if not missing:
        return
    if not allow_download:
        validate_seedvr2_checkpoints(ckpts_root=ckpts_root, variant=variant)
        return

    try:
        hf_hub_download = _load_hf_hub_download()
    except ImportError as exc:
        raise RuntimeError(
            "allow_checkpoint_download=true requires the optional "
            "'huggingface_hub' package to be installed. Install super-resolution[seedvr2] "
            "or use the SeedVR2 service container."
        ) from exc

    token = os.getenv("HF_TOKEN") or None
    repo_by_variant = {
        "seedvr2_3b": os.getenv("HF_REPO_SEEDVR2_3B") or "ByteDance-Seed/SeedVR2-3B",
        "seedvr2_7b": os.getenv("HF_REPO_SEEDVR2_7B") or "ByteDance-Seed/SeedVR2-7B",
    }
    fallback_repos = [repo_by_variant["seedvr2_7b"], repo_by_variant["seedvr2_3b"]]
    target_repo = repo_by_variant[str(variant)]
    ckpts_dir = Path(ckpts_root) / "seedvr2"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    for path in missing:
        repos = fallback_repos if path.name == "ema_vae.pth" else [target_repo]
        last_error: Exception | None = None
        for repo_id in repos:
            try:
                logger.info("Downloading SeedVR2 checkpoint %s:%s", repo_id, path.name)
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=path.name,
                    token=token,
                    local_dir=str(ckpts_dir),
                )
                downloaded_path = Path(downloaded)
                if downloaded_path.resolve() != path.resolve():
                    downloaded_path.replace(path)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"Failed to download SeedVR2 checkpoint: {path.name}"
            ) from last_error

    validate_seedvr2_checkpoints(ckpts_root=ckpts_root, variant=variant)


def _missing_checkpoints(*, ckpts_root: Path, variant: str) -> list[Path]:
    ckpts_dir = Path(ckpts_root).expanduser().resolve() / "seedvr2"
    required = _VARIANT_CHECKPOINTS[str(variant)]
    return [
        ckpts_dir / filename for filename in required if not _is_nonempty_file(ckpts_dir / filename)
    ]


def _clean_optional_path(value: str | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "none":
        return None
    return Path(raw).expanduser().resolve()


def _all_gpu_ids() -> list[int]:
    try:
        torch = importlib.import_module("torch")
        count = int(torch.cuda.device_count()) if bool(torch.cuda.is_available()) else 0
    except Exception as exc:
        raise RuntimeError(_NO_CUDA_DEVICES_MESSAGE) from exc
    if count <= 0:
        raise RuntimeError(_NO_CUDA_DEVICES_MESSAGE)
    return list(range(count))


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _runtime_pythonpath(seedvr_root: Path) -> str:
    entries = [str(seedvr_root)]
    entries.extend(path for path in sys.path if path and "site-packages" in path)
    existing = os.getenv("PYTHONPATH")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)
    return os.pathsep.join(dict.fromkeys(entries))


def _remove_existing_output(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _ensure_runtime_symlink(*, link_path: Path, target: Path, description: str) -> None:
    target = Path(target).expanduser().resolve()
    link_path = Path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        if _runtime_link_matches(link_path=link_path, target=target):
            return
        _remove_runtime_link_conflict(link_path=link_path, description=description)
        try:
            link_path.symlink_to(target, target_is_directory=target.is_dir())
            return
        except FileExistsError:
            if _runtime_link_matches(link_path=link_path, target=target):
                return
            continue
        except OSError as exc:
            if _is_missing_path_error(exc):
                continue
            raise
    if _runtime_link_matches(link_path=link_path, target=target):
        return
    raise RuntimeError(f"Cannot create {description} symlink at {link_path}; path already exists.")


def _remove_runtime_link_conflict(*, link_path: Path, description: str) -> None:
    try:
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            if link_path.is_dir() and not any(link_path.iterdir()):
                link_path.rmdir()
            else:
                raise RuntimeError(
                    f"Cannot create {description} symlink at {link_path}; path already exists."
                )
    except OSError as exc:
        if _is_missing_path_error(exc):
            return
        raise


def _runtime_link_matches(*, link_path: Path, target: Path) -> bool:
    try:
        return link_path.is_symlink() and link_path.resolve(strict=False) == target
    except OSError:
        return False


def _is_missing_path_error(exc: OSError) -> bool:
    return isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT


def _load_hf_hub_download() -> Callable[..., str]:
    module = importlib.import_module("huggingface_hub")
    return cast(Callable[..., str], module.hf_hub_download)


__all__ = [
    "CommandInvocation",
    "CommandRunnerError",
    "CommandRunner",
    "SeedVR2Resolver",
    "SubprocessCommandRunner",
    "ensure_seedvr2_checkpoints",
    "prepare_seedvr_runtime_root",
    "resolve_gpu_ids",
    "resolve_model_cache_root",
    "resolve_seedvr_source_root",
    "validate_seedvr_source_root",
    "validate_seedvr2_runtime_command",
    "validate_seedvr2_checkpoints",
]
