# VIDEO JSON PROMPT

You are given sampled frames from one video. The frames may come from the
original input, a super-resolution output, or a tracking overlay with visible
instance IDs. Describe the overall video scene, visible context, and main
activity without producing individual events.

## OUTPUT

Return EXACTLY ONE fenced JSON block, and nothing else:

```json
{
  "scene_description": "<concise description of the scene and visible actors/objects>",
  "event_summary": "<concise summary of visible activity, or 'No meaningful activity or incident is visible.'>",
  "scenario_info": "<short phrase naming the scene type>"
}
```

## RULES

- Output a single JSON object for video.json only. Do NOT include an `events`
  key in this response.
- Use only visible evidence from the frames. Do not infer hidden causes or
  unseen outcomes.
- If the input is an overlay with visible IDs, you may refer to those IDs in
  prose when useful.
- Keep the prompt domain-neutral. Describe only the visible scene, subjects,
  objects, setting, and activity; do not assume a specialized domain.
- Keep text concise and factual:
  - Do not list every frame or every visible object.
  - Keep each text field to one short sentence or phrase.
  - Omit optional fields when they would add low-confidence or repetitive detail.
