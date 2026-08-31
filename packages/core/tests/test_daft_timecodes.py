# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from core.formats.daft.timecodes import seconds_to_timecode


def test_seconds_to_timecode_rejects_hours_outside_schema() -> None:
    assert seconds_to_timecode(359999.999) == "99:59:59.999"

    with pytest.raises(ValueError, match="exceeds DAFT timecode range"):
        seconds_to_timecode(360000)
