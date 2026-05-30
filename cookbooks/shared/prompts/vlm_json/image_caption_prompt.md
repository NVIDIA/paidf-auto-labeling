# IMAGE JSON PROMPT

You are given ONE still image. There is no motion, no time axis, and nothing
happens after the frame ends. Your job is to produce a concise factual caption
describing what is visible in this frame for image.json.

## OUTPUT

Return EXACTLY ONE fenced JSON block, and nothing else:

```json
{
  "caption": "<one or two sentences in plain prose, describing what is visible>",
  "scenario_info": "<short phrase naming the scene type; omit this key entirely if uncertain>"
}
```

`scenario_info` should be omitted entirely (do not include the key) when you are not confident about the scene type.

## RULES

- Output exactly ONE fenced ```json``` block. No prose, headers, or commentary
  outside the block.
- `caption` is REQUIRED. Keep it factual and grounded in the pixels — do not
  invent events, motion, time of day, weather, or actors that are not visible.
- `scenario_info` is OPTIONAL. Omit it (do not emit the key) if you are not
  confident.
- For unclear, corrupted, or empty images, emit a minimal factual `caption`
  describing only what is visible, such as "blurred image" or "empty frame",
  and omit the optional `scenario_info` key.
- Keep the prompt domain-neutral. Describe only the visible scene, subjects,
  objects, setting, and activity; do not assume a specialized domain.
- Do NOT emit any of these fields: `events`, `event_summary`,
  `scene_description`, `fps`, `duration`, `video_id`, `image_id`. They do not
  apply to a still image and will be discarded.
- Do NOT copy any example IDs, tracking numbers, or category labels from prior
  prompts you may have seen. Describe only what is in THIS image.
- Keep the response compact:
  - Do not list every visible object.
  - Use two sentences maximum for `caption`.
  - Keep `scenario_info` to a short phrase.
  - Omit optional fields when they would add low-confidence or repetitive detail.
