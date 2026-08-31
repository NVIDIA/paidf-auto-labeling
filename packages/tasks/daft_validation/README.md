# DAFT Validation

`daft-validation` provides a strict structural validation task for DAFT scene
outputs. It is intended to run at the end of service pipelines after annotation
tasks have written their DAFT files.

The task validates canonical JSON files under `contextual/` and `task/`, rejects
unknown annotation filenames in those directories, and checks that scene media
IDs match the current `DataEntry`.
