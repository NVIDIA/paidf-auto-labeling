# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Exceptions that task implementations can raise to communicate task-specific failures.

These exception types are intended for errors encountered while a task is processing an
individual data entry. They are not expected to be used outside a task.
"""


class RetryTaskError(Exception):
    """
    Signal that a task encountered a transient failure for the current data entry.

    Task implementations should raise this exception when retrying the same task against the same
    data entry may succeed without changing the input. Examples include temporary network failures,
    rate limits, unavailable model servers, or other short-lived infrastructure errors.

    Pipeline implementations may choose to catch this exception and retry the current task/data
    entry pair according to their retry policy. Until a pipeline explicitly implements that policy,
    this exception behaves like any other exception and propagates to the caller.
    """


class InvalidInputError(Exception):
    """
    Signal that the current data entry is invalid for a task and should be excluded.

    Task implementations should raise this exception when the task cannot process a specific data
    entry because the entry's content is invalid, unsupported, corrupt, or otherwise unsuitable for
    annotation. This is intended for per-entry data problems, not fatal task configuration errors
    or framework setup failures.

    Pipeline implementations may choose to catch this exception and omit the current data entry
    from downstream tasks and final results. Until a pipeline explicitly implements that behavior,
    this exception behaves like any other exception and propagates to the caller.
    """
