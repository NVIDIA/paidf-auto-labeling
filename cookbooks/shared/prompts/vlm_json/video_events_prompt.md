# VIDEO EVENTS PROMPT

You are given timestamped frames from one video. The frames may come from the
original input, a super-resolution output, or an overlay with visible instance
IDs. Identify concise, visible events from the sampled frames.

An event is any event-worthy visual content over a time span. It may be:

- an action or movement
- an interaction
- a visible state or sustained condition
- a visible arrangement
- an incident or anomaly

## OUTPUT

Return EXACTLY ONE fenced JSON block, and nothing else:

```json
{
  "events": []
}
```

When visible events exist, each event may include:

```json
{
  "events": [
    {
      "event_id": "event_001",
      "event_caption": "<one sentence describing the visible event>",
      "start_time": 0.0,
      "end_time": 1.0,
      "category": "<one allowed category>",
      "instances": []
    }
  ]
}
```

## RULES

### Event Selection

- `events` is REQUIRED and must be a list.
- Return at least one event when any event-worthy action, interaction, state,
  sustained condition, arrangement, incident, or anomaly is visible.
- Return `{"events": []}` only when the sampled frames show no event-worthy
  visual content at all.
- Prefer 2-5 events when the video clearly contains multiple distinct visible
  states, activities, conditions, arrangements, incidents, or anomalies.
- Merge repeated, continuous, or near-duplicate content into one aggregated
  event covering the full visible time span.
- If several subjects perform the same kind of motion or activity, summarize
  them as one group event unless one subject is clearly the focus.
- Pay special attention to interactions and spatial relationships between
  subjects or objects, such as approaching, crossing paths, blocking, yielding,
  avoiding, contacting, or moving near each other.
- Avoid repeated caption templates. Each event should describe a distinct
  visible state, activity, direction, interaction, condition, or incident.
- Do NOT create one event per frame or per second.
- Use only visible evidence from the frames. Do not infer hidden causes,
  identities, intentions, or outcomes.

### Categories And Time

- Use these allowed `category` values only: `movement`, `interaction`,
  `state_change`, `activity`, `incident`, `anomaly`, `other`.
- Choose the most specific allowed category that is directly supported by the
  frames:
  - `movement`: visible motion
  - `interaction`: visible subjects or objects affecting each other
  - `state_change`: discrete transitions or sustained conditions not explained
    by motion or intentional activity, such as an object opened/closed, a light
    turning on/off, a person sitting/standing, or a pose held for more than 2
    seconds. Do not use this for transient motion or purposeful activity.
  - `activity`: purposeful visible activity
  - `incident` / `anomaly`: unsafe, abnormal, unexpected, or impact-like content
    with concrete visible evidence. Examples include subjects on a collision
    course or within immediate interaction distance, collisions, falls, abrupt
    stops, obstructions blocking egress, or sudden deviations from the baseline
    behavior visible in the clip.
  - `other`: event-worthy content only when none of the tightened definitions
    above apply
- Use timestamps visible in the prompt to estimate `start_time` and `end_time`
  in seconds. These fields are required for every video event.

### Instances

- `instances` is optional grounding, not an inventory of everything visible.
- Include IDs only for clearly labeled subjects that are directly named by the
  event caption.
- If the caption describes one clearly labeled subject, include exactly one ID
  for that subject.
- If the caption describes two or three clearly labeled subjects, include only
  those exact IDs.
- If the caption describes a group, crowd, collection, flow, repeated motion,
  dense scene, or many similar subjects, use `instances: []`.
- Do NOT list all visible IDs. Do NOT include nearby, unrelated, background, or
  merely co-visible IDs.
- Do NOT include an ID unless the readable label is visibly attached to the same
  subject described by the caption.
- Do NOT include an ID if the subject type, appearance, or role is uncertain or
  conflicts with the caption.
- Do NOT infer IDs from position, class, continuity, or context.
- When including an ID, write it as `id_<number>` using the readable overlay
  number. Do NOT write bare numbers such as `"2"` and do NOT invent DAFT keys
  such as `"person_2"`.
- Keep the full response compact: at most 5 events and at most 3 instance IDs
  per event.

### Compactness

- Return at most 5 events total. This is a hard limit.
- Keep each `event_caption` to one short sentence.
- Keep the full JSON compact. If uncertain, return fewer events and fewer IDs.
- Omit optional fields when they would add low-confidence or repetitive detail.
