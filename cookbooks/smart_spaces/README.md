# Smart Spaces

Smart Spaces cookbooks focus on indoor and operational environments such as
warehouses, facilities, and monitored work areas. The first checked-in scenario
uses the warehouse sample video for event verification and reasoning.

## Warehouse Event Reasoning

```text
detection_and_tracking -> captioning -> event_verification_visual_qa -> reasoning
```

This scenario verifies warehouse safety and operational events, writes Visual QA
evidence, then runs reasoning tasks such as event summaries, open QA,
closed-choice QA, causal linkage, and temporal localization.

Run a dry plan from a fresh checkout:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/smart_spaces/configs/pipeline_warehouse_event_reasoning.yaml --container-dry-run'
```

For real runs, copy the config to `pipeline_warehouse_event_reasoning.local.yaml`
and replace the SAM3 mount, output path, endpoint URLs, and served model names.
Keep credentials out of the config and pass only environment variable names such
as `NVIDIA_API_KEY`.
