# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow direct invocation without PYTHONPATH being set
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from al_utils.ckpts import ensure_url_downloaded, resolve_ckpts_root
from detection_and_tracking.rfdetr_tracking import RFDETR_PRETRAIN_URLS, ensure_reid_weights
from sr_runner.seedvr2 import ensure_seedvr2_ckpts


def _repo_root() -> Path:
    # modules/al_utils/ -> modules/ -> repo/
    return Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prefetch model checkpoints into ckpts_root.")
    ap.add_argument(
        "--ckpts-root",
        default="",
        help="Override ckpts root directory (otherwise uses MODEL_CACHE_PATH / defaults).",
    )
    ap.add_argument(
        "--sr-variant",
        default="",
        help="Prefetch SeedVR2 checkpoints for this variant (seedvr2_3b or seedvr2_7b).",
    )
    ap.add_argument("--with-reid", action="store_true", help="Prefetch ReID weights (clip_vehicleid.pt).")
    ap.add_argument("--with-rfdetr", action="store_true", help="Prefetch RF-DETR weights (rf-detr-base.pth).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(levelname)s: %(message)s")
    logger = logging.getLogger("prefetch")

    root = _repo_root()
    ckpts_root = (
        Path(str(args.ckpts_root)).expanduser().resolve()
        if str(args.ckpts_root).strip()
        else resolve_ckpts_root(repo_root=root)
    )
    logger.info("ckpts_root=%s", ckpts_root)

    if str(args.sr_variant).strip():
        ensure_seedvr2_ckpts(ckpts_root=ckpts_root, variant=str(args.sr_variant).strip(), logger=logger)

    if bool(args.with_rfdetr):
        pretrain_name = "rf-detr-base.pth"
        url = RFDETR_PRETRAIN_URLS[pretrain_name]
        dst = ckpts_root / "rfdetr" / pretrain_name
        logger.info("prefetch rfdetr: %s", dst)
        ensure_url_downloaded(url=url, dst=dst, timeout_s=600)

    if bool(args.with_reid):
        dst = ckpts_root / "reid" / "clip_vehicleid.pt"
        logger.info("prefetch reid: %s", dst)
        ensure_reid_weights(dst, logger=logger)

    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
