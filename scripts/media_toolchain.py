#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install and verify the repository's controlled media toolchain."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

import ffmpeg_codec_policy

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "media-toolchain.toml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"Manifest section [{name}] is missing or invalid.")
    return value


def _string(section: dict[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Manifest value [{section_name}].{key} is missing or invalid.")
    return value


def _requirements(section: dict[str, Any], section_name: str) -> list[str]:
    value = section.get("build_requirements")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(
            f"Manifest value [{section_name}].build_requirements is missing or invalid."
        )
    return value


def _run_script(script: str, arguments: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        ["bash", str(SCRIPT_DIR / script), *arguments],
        check=True,
        env={**os.environ, **env},
    )


def _ffmpeg_install(args: argparse.Namespace, config: dict[str, Any]) -> int:
    ffmpeg = _section(config, "ffmpeg")
    headers = _section(config, "nv_codec_headers")
    libvpx = _section(config, "libvpx")
    _run_script(
        "build_restricted_ffmpeg.sh",
        [
            args.profile,
            str(SCRIPT_DIR / "ffmpeg_codec_policy.py"),
            str(args.artifact_root),
            str(args.prefix),
            str(args.source_dir),
        ],
        {
            "FFMPEG_VERSION": _string(ffmpeg, "version", "ffmpeg"),
            "FFMPEG_URL": _string(ffmpeg, "url", "ffmpeg"),
            "FFMPEG_SHA256": _string(ffmpeg, "sha256", "ffmpeg"),
            "NV_CODEC_HEADERS_VERSION": _string(headers, "version", "nv_codec_headers"),
            "NV_CODEC_HEADERS_URL": _string(headers, "url", "nv_codec_headers"),
            "NV_CODEC_HEADERS_SHA256": _string(headers, "sha256", "nv_codec_headers"),
            "LIBVPX_VERSION": _string(libvpx, "version", "libvpx"),
            "LIBVPX_URL": _string(libvpx, "url", "libvpx"),
            "LIBVPX_SHA256": _string(libvpx, "sha256", "libvpx"),
            "MEDIA_BUILD_JOBS": str(args.jobs),
        },
    )
    return 0


def _python_install(args: argparse.Namespace, config: dict[str, Any]) -> int:
    for component in args.component:
        if component == "pyav":
            section_name = "pyav"
            script = "install_pyav_from_source.sh"
            version_key = "PYAV_VERSION"
            checksum_key = "PYAV_SHA256"
        else:
            section_name = "opencv_headless"
            script = "install_opencv_headless_from_source.sh"
            version_key = "OPENCV_PYTHON_HEADLESS_VERSION"
            checksum_key = "OPENCV_PYTHON_HEADLESS_SHA256"
        section = _section(config, section_name)
        env = {
            version_key: _string(section, "version", section_name),
            checksum_key: _string(section, "sha256", section_name),
            "MEDIA_BUILD_REQUIREMENTS": "\n".join(_requirements(section, section_name)),
        }
        if args.wheel_dir is not None:
            env["MEDIA_WHEEL_DIR"] = str(args.wheel_dir)
        _run_script(
            script,
            [
                str(args.python),
                str(args.artifact_root),
                str(args.source_dir),
                str(args.prefix),
            ],
            env,
        )
    return 0


def _target_site_packages(python: Path) -> Path:
    if python.resolve() == Path(sys.executable).resolve():
        return Path(sysconfig.get_path("purelib"))
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip())


def _verify(args: argparse.Namespace, config: dict[str, Any]) -> int:
    del config
    roots = list(args.site_packages)
    if args.python is not None:
        roots.append(_target_site_packages(args.python))
    if not roots:
        roots = [Path(sysconfig.get_path("purelib"))]
    return int(
        ffmpeg_codec_policy.verify_media_policy(
            profile=args.profile,
            roots=roots,
            ffmpeg_executable=str(args.ffmpeg),
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ffmpeg = subparsers.add_parser("ffmpeg-install", help="Build the controlled FFmpeg.")
    ffmpeg.add_argument("--profile", choices=("input-only", "vp9-output"), required=True)
    ffmpeg.add_argument("--artifact-root", type=Path, default=Path("/opt/media-runtime"))
    ffmpeg.add_argument("--prefix", type=Path, default=Path("/usr/local"))
    ffmpeg.add_argument("--source-dir", type=Path, default=Path("/usr/share/oss-sources"))
    ffmpeg.add_argument("--jobs", type=_positive_int, default=os.cpu_count() or 1)
    ffmpeg.set_defaults(handler=_ffmpeg_install)

    python_install = subparsers.add_parser(
        "python-install", help="Build controlled Python media packages from source."
    )
    python_install.add_argument("--python", type=Path, required=True)
    python_install.add_argument(
        "--component",
        action="append",
        choices=("pyav", "opencv-headless"),
        required=True,
    )
    python_install.add_argument("--artifact-root", type=Path, default=Path("/opt/media-runtime"))
    python_install.add_argument("--source-dir", type=Path, default=Path("/usr/share/oss-sources"))
    python_install.add_argument("--prefix", type=Path, default=Path("/usr/local"))
    python_install.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Also emit the built component as a wheel into this directory (for a shared base).",
    )
    python_install.set_defaults(handler=_python_install)

    verify = subparsers.add_parser("verify", help="Verify the installed codec surface.")
    verify.add_argument("--profile", choices=("input-only", "vp9-output"), required=True)
    verify.add_argument("--python", type=Path)
    verify.add_argument("--site-packages", type=Path, action="append", default=[])
    verify.add_argument("--ffmpeg", type=Path, default=Path("/usr/local/bin/ffmpeg"))
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _manifest(args.manifest)
    return int(args.handler(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
