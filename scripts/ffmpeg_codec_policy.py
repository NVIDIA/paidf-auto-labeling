#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define and validate the approved FFmpeg codec surface for container builds.

The ``configure-flags`` command is an internal build-script interface. The
public container-build interface is ``media_toolchain.py``.

This is a focused image-build compliance gate, not a comprehensive license
scanner. It detects Python wheels containing private FFmpeg/libav payloads and
FFmpeg builds containing common GPL, nonfree, H.264, or H.265 software codec
markers.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

COMMON_INPUT_FLAGS: tuple[str, ...] = (
    "--enable-decoder=h264_cuvid",
    "--enable-decoder=vp9_cuvid",
    "--enable-decoder=vp9",
    "--enable-decoder=mpeg4",
    "--enable-nvdec",
    "--enable-protocol=file",
    "--enable-protocol=pipe",
    "--enable-demuxer=mov",
    "--enable-demuxer=matroska",
    "--enable-parser=h264",
    "--enable-parser=vp9",
    "--enable-parser=mpeg4video",
    "--enable-bsf=h264_mp4toannexb",
    "--enable-filter=scale",
    # ``select`` windows frames during decode (core.media window streaming) and
    # in the VLM window re-encoder; ``setpts`` normalises timestamps on the
    # re-encoded window. Both must be explicitly enabled under
    # ``--disable-everything``.
    "--enable-filter=select",
    "--enable-filter=setpts",
)
OUTPUT_PROFILE_FLAGS: dict[str, tuple[str, ...]] = {
    "input-only": ("--enable-encoder=rawvideo", "--enable-muxer=rawvideo"),
    "vp9-output": (
        "--enable-libvpx",
        "--enable-encoder=libvpx_vp9",
        "--enable-encoder=rawvideo",
        "--enable-muxer=mp4",
        "--enable-muxer=mov",
        "--enable-muxer=rawvideo",
    ),
}

_FFMPEG_LIB_PREFIXES = (
    "libavcodec",
    "libavdevice",
    "libavfilter",
    "libavformat",
    "libavutil",
    "libpostproc",
    "libswresample",
    "libswscale",
)
_FORBIDDEN_CODEC_MARKERS = {
    b"--enable-gpl": "--enable-gpl",
    b"--enable-nonfree": "--enable-nonfree",
    b"libopenh264": "libopenh264",
    b"libx264": "libx264",
    b"libx265": "libx265",
}
_FORBIDDEN_FFMPEG_LICENSE_MARKERS = {
    "gnu general public license": "GPL license text",
    "nonfree and unredistributable": "nonfree/unredistributable license text",
    "not legally redistributable": "nonfree redistribution warning",
}
_FORBIDDEN_FFMPEG_BUILDCONF_MARKERS = {
    "--enable-gpl": "--enable-gpl",
    "--enable-nonfree": "--enable-nonfree",
    "--enable-libaacplus": "--enable-libaacplus",
    "--enable-libfdk-aac": "--enable-libfdk-aac",
    "--enable-libopenh264": "--enable-libopenh264",
    "--enable-libx264": "--enable-libx264",
    "--enable-libx265": "--enable-libx265",
}
_FORBIDDEN_FFMPEG_COMPONENT_PREFIXES = (
    "aac",
    "hevc",
    "h265",
    "libfdk_aac",
    "libx265",
)
# FFmpeg 8.1's MPEG-4 Part 2 decoder selects and exposes the H.263 decoder as a
# build dependency. H.263 remains an unsupported input codec: core.media only
# selects MPEG-4, VP9, and the approved H.264 hardware decoder.
_COMMON_INPUT_DECODERS = {"h264_cuvid", "mpeg4", "vp9", "vp9_cuvid"}
_FFMPEG_DECODER_ALLOWLIST = _COMMON_INPUT_DECODERS | {"h263"}
# Rawvideo supports decoder-to-Python frame pipes. Persisted output is encoded
# by libvpx-vp9 when the vp9-output profile is selected.
_OUTPUT_PROFILE_ENCODERS = {
    "input-only": {"rawvideo"},
    "vp9-output": {"libvpx-vp9", "rawvideo"},
}
_ALLOWED_BSFS = {
    "aac_adtstoasc",
    "h264_mp4toannexb",
    "vp9_superframe",
    "vp9_superframe_split",
}
_EXPECTED_FFMPEG_COMPONENTS = {
    "encoders": _OUTPUT_PROFILE_ENCODERS["vp9-output"],
    "decoders": _FFMPEG_DECODER_ALLOWLIST,
    "bitstream filters": _ALLOWED_BSFS,
}


def expected_ffmpeg_components(profile: str) -> dict[str, set[str]]:
    """Return common input requirements plus one service output allowlist."""
    try:
        encoders = _OUTPUT_PROFILE_ENCODERS[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(_OUTPUT_PROFILE_ENCODERS))
        raise ValueError(f"Unknown codec output profile {profile!r}; choices: {choices}") from exc
    return {
        "encoders": set(encoders),
        "decoders": set(_FFMPEG_DECODER_ALLOWLIST),
        "bitstream filters": set(_ALLOWED_BSFS),
    }


def flags_for_profile(profile: str) -> tuple[str, ...]:
    """Return common approved input flags plus one service output profile."""
    try:
        output_flags = OUTPUT_PROFILE_FLAGS[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(OUTPUT_PROFILE_FLAGS))
        raise ValueError(f"Unknown codec output profile {profile!r}; choices: {choices}") from exc
    return COMMON_INPUT_FLAGS + output_flags


def main(argv: list[str]) -> int:
    """Print configure flags for the internal FFmpeg build script."""
    if len(argv) != 3 or argv[1] != "configure-flags":
        print(f"usage: {argv[0]} configure-flags <profile>", file=sys.stderr)
        return 2
    try:
        flags = flags_for_profile(argv[2])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(*flags, sep="\n")
    return 0


def verify_media_policy(
    *,
    profile: str,
    roots: list[Path],
    ffmpeg_executable: str,
) -> int:
    """Check Python package roots and one required FFmpeg executable."""
    expected_components = expected_ffmpeg_components(profile)
    roots = _existing_roots(roots)
    bundled_libs = sorted(
        path for root in roots for path in root.rglob("*") if _is_forbidden_bundled_lib(path)
    )
    bundled_binaries = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if _is_forbidden_bundled_ffmpeg_binary(path)
    )
    codec_findings = _find_forbidden_codec_markers(roots)
    ffmpeg_findings = _find_forbidden_ffmpeg_cli_markers(
        ffmpeg_executable,
        expected_components,
    )

    if not bundled_libs and not bundled_binaries and not codec_findings and not ffmpeg_findings:
        rendered = ", ".join(str(root) for root in roots)
        print(f"Media codec policy passed for: {rendered}")
        return 0

    print("ERROR: media codec policy failed.", file=sys.stderr)
    print(
        "This image must not ship upstream wheel-bundled FFmpeg/libav payloads "
        "or FFmpeg builds containing GPL/nonfree/H.264/H.265 software codec markers.",
        file=sys.stderr,
    )
    if bundled_libs:
        print("\nBundled FFmpeg/libav shared libraries found:", file=sys.stderr)
        for path in bundled_libs:
            print(f"  {path}", file=sys.stderr)
    if bundled_binaries:
        print("\nBundled FFmpeg executable payloads found:", file=sys.stderr)
        for path in bundled_binaries:
            print(f"  {path}", file=sys.stderr)
    if codec_findings:
        print("\nForbidden codec/config markers found:", file=sys.stderr)
        for path, markers in codec_findings:
            print(f"  {path}: {', '.join(markers)}", file=sys.stderr)
    if ffmpeg_findings:
        print("\nForbidden ffmpeg binary markers found:", file=sys.stderr)
        for finding in ffmpeg_findings:
            print(f"  {finding}", file=sys.stderr)
    return 1


def _existing_roots(roots: list[Path]) -> list[Path]:
    roots = [root.resolve() for root in roots]
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise SystemExit("No existing site-packages roots found to scan.")
    return existing_roots


def _is_forbidden_bundled_lib(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.name.startswith(_FFMPEG_LIB_PREFIXES)


def _is_forbidden_bundled_ffmpeg_binary(path: Path) -> bool:
    if not path.is_file():
        return False
    if not path.name.startswith("ffmpeg"):
        return False
    return path.suffix not in {".py", ".pyi", ".typed", ".txt", ".md"}


def _find_forbidden_codec_markers(roots: list[Path]) -> list[tuple[Path, list[str]]]:
    findings: list[tuple[Path, list[str]]] = []
    for root in roots:
        for path in sorted(root.rglob("libavcodec*.so*")):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            markers = [
                label for marker, label in _FORBIDDEN_CODEC_MARKERS.items() if marker in payload
            ]
            if markers:
                findings.append((path, markers))
    return findings


def _find_forbidden_ffmpeg_cli_markers(
    ffmpeg_executable: str,
    expected_components: dict[str, set[str]] | None = None,
) -> list[str]:
    ffmpeg_path = _resolve_executable(ffmpeg_executable)
    if ffmpeg_path is None:
        return [f"required FFmpeg executable not found: {ffmpeg_executable}"]

    findings: list[str] = []
    license_output = _run_ffmpeg_probe(ffmpeg_path, "-L")
    if license_output is None:
        findings.append(f"{ffmpeg_path}: could not run `ffmpeg -L`")
    else:
        findings.extend(
            _matching_text_markers(
                ffmpeg_path,
                "ffmpeg -L",
                license_output,
                _FORBIDDEN_FFMPEG_LICENSE_MARKERS,
            )
        )

    buildconf_output = _run_ffmpeg_probe(ffmpeg_path, "-buildconf")
    if buildconf_output is None:
        findings.append(f"{ffmpeg_path}: could not run `ffmpeg -buildconf`")
    else:
        findings.extend(
            _matching_text_markers(
                ffmpeg_path,
                "ffmpeg -buildconf",
                buildconf_output,
                _FORBIDDEN_FFMPEG_BUILDCONF_MARKERS,
            )
        )

    findings.extend(_find_forbidden_ffmpeg_components(ffmpeg_path, expected_components))
    return findings


def _resolve_executable(executable: str) -> str | None:
    if "/" in executable:
        path = Path(executable)
        return str(path.resolve()) if path.is_file() else None
    return shutil.which(executable)


def _run_ffmpeg_probe(ffmpeg_path: str, option: str) -> str | None:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", option],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _matching_text_markers(
    executable: str,
    command_label: str,
    output: str,
    markers: dict[str, str],
) -> list[str]:
    lowered = output.lower()
    return [
        f"{executable}: {command_label} contains {label}"
        for marker, label in markers.items()
        if marker in lowered
    ]


def _find_forbidden_ffmpeg_components(
    ffmpeg_path: str,
    expected_components: dict[str, set[str]] | None = None,
) -> list[str]:
    expected = expected_components or _EXPECTED_FFMPEG_COMPONENTS
    findings: list[str] = []
    component_specs = (
        ("encoders", "-encoders", 1, expected["encoders"]),
        ("decoders", "-decoders", 1, expected["decoders"]),
        ("bitstream filters", "-bsfs", 0, _ALLOWED_BSFS),
    )
    for label, option, name_index, allowed_components in component_specs:
        output = _run_ffmpeg_probe(ffmpeg_path, option)
        if output is None:
            findings.append(f"{ffmpeg_path}: could not run `ffmpeg {option}`")
            continue
        component_names = _ffmpeg_component_names(output, name_index)
        forbidden_components = _forbidden_ffmpeg_component_names(
            component_names,
            allowed_components=allowed_components,
        )
        if forbidden_components:
            findings.append(
                f"{ffmpeg_path}: ffmpeg {option} exposes forbidden {label}: "
                f"{', '.join(forbidden_components)}"
            )
        unexpected_components = sorted(set(component_names) - expected[label])
        if unexpected_components:
            findings.append(
                f"{ffmpeg_path}: ffmpeg {option} exposes unexpected {label}: "
                f"{', '.join(unexpected_components)}"
            )
    return findings


def _ffmpeg_component_names(output: str, name_index: int) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) <= name_index:
            continue
        name = parts[name_index]
        if (
            name.startswith("-")
            or name == "="
            or name.endswith(":")
            or any(part.endswith(":") for part in parts)
        ):
            continue
        names.append(name)
    return names


def _forbidden_ffmpeg_component_names(
    component_names: list[str],
    *,
    allowed_components: set[str],
) -> list[str]:
    forbidden: list[str] = []
    for name in component_names:
        if name in allowed_components:
            continue
        if name.startswith(_FORBIDDEN_FFMPEG_COMPONENT_PREFIXES):
            forbidden.append(name)
            continue
        if name.startswith("h264") or name in {"libopenh264", "libx264"}:
            forbidden.append(name)
    return sorted(set(forbidden))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
