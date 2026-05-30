# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration test: MSC upload and download (real network I/O).

Run only when MULTISTORAGECLIENT_CONFIGURATION and MSC_TEST_REMOTE_PATH are set.
Use an existing bucket/path where you have read and write (and ideally delete) permission.
If you only have read access, upload will fail; if you have write but not delete, the test
leaves the file and prints a warning—remove test_upload_download.txt manually if needed.

Example (S3 with env creds; path_mapping optional for plain s3://):
  source ~/aws.sh   # or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, etc.
  export MULTISTORAGECLIENT_CONFIGURATION='{}'
  export MSC_TEST_REMOTE_PATH='s3://YOUR-EXISTING-BUCKET/msc-integration-test/'
  python tests/test_msc_sync_integration.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import multistorageclient as msc
from nvcf_msc_utils import (
    setup_msc_config,
    sync_local_to_remote,
    sync_remote_to_local,
)

TEST_FILENAME = "test_upload_download.txt"
# Subdir under MSC_TEST_REMOTE_PATH so we only upload/download one file (no sync of existing content)
TEST_SUBDIR = "msc_integration_test"


def _is_permission_error(e: Exception) -> bool:
    msg = str(e).lower()
    if "access denied" in msg or "403" in msg or "forbidden" in msg or "permission" in msg:
        return True
    err = getattr(e, "response", None)
    if isinstance(err, dict):
        code = (err.get("Error") or {}).get("Code", "")
        return str(code) in ("AccessDenied", "Forbidden")
    return False


def main() -> int:
    config = os.environ.get("MULTISTORAGECLIENT_CONFIGURATION", "").strip()
    remote_path = os.environ.get("MSC_TEST_REMOTE_PATH", "").strip().rstrip("/")
    if not config or not remote_path:
        print(
            "SKIP: set MULTISTORAGECLIENT_CONFIGURATION and MSC_TEST_REMOTE_PATH (e.g. s3://bucket/msc-test/) to run upload/download test."
        )
        return 0

    # Use a dedicated subdir so we only touch one file; no sync of existing content
    test_prefix = remote_path.rstrip("/") + "/" + TEST_SUBDIR + "/"
    setup_msc_config()

    test_content = b"hello msc upload download test\n"
    with tempfile.TemporaryDirectory(prefix="msc_sync_up_") as upload_dir:
        test_file = Path(upload_dir) / TEST_FILENAME
        test_file.write_bytes(test_content)

        print("Upload: local -> remote (single file)")
        try:
            sync_local_to_remote(upload_dir, test_prefix, verbose=True)
        except FileNotFoundError as e:
            if "does not exist" in str(e) and "404" in str(e):
                print("Hint: bucket or path may not exist; set MSC_TEST_REMOTE_PATH to an existing bucket/path.")
            raise
        except Exception as e:
            if _is_permission_error(e):
                print("Hint: no write permission? Use a path where you have put/delete access, or check IAM/policy.")
            raise

    time.sleep(1)

    with tempfile.TemporaryDirectory(prefix="msc_sync_down_") as download_dir:
        print("Download: remote -> local (single file)")
        sync_remote_to_local(test_prefix, download_dir, verbose=True)

        downloaded = Path(download_dir) / TEST_FILENAME
        if not downloaded.exists():
            print(f"FAIL: downloaded file not found: {downloaded}")
            return 1
        if downloaded.read_bytes() != test_content:
            print("FAIL: downloaded content does not match")
            return 1

    # Restore: remove the test file from remote
    remote_test_file = test_prefix + TEST_FILENAME
    try:
        msc.delete(remote_test_file)
        print(f"Restored: removed remote {TEST_FILENAME}")
    except Exception as e:
        print(f"Warning: could not remove remote test file ({remote_test_file}): {e}")
        if _is_permission_error(e):
            print(f"Hint: no delete permission? Remove {TEST_FILENAME} manually from the path if needed.")

    print("OK: upload and download verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
