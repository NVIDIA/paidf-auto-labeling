# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from string import ascii_lowercase, ascii_uppercase

# DAFT's MCQ answer regex is ^[A-Za-z]$, giving 52 distinct letter keys.
LETTER_ALPHABET: tuple[str, ...] = tuple(ascii_uppercase + ascii_lowercase)

# Bank-authored choices may use period, closing-parenthesis, or colon prefixes,
# for example "A. Rollover", "B) Head-on", or "C: Rear-end".
LETTER_PREFIX_RE = re.compile(r"^\s*([A-Za-z])[.):]\s*(.*)$", re.DOTALL)


def match_letter_prefix(text: str) -> re.Match[str] | None:
    return LETTER_PREFIX_RE.match(text)


def strip_letter_prefix(text: str) -> str:
    match = match_letter_prefix(text)
    return match.group(2) if match else text
