# Traffic Accident Understanding

You are analyzing traffic surveillance footage to produce a **structured scene description** that will later be mapped to the fixed multiple-choice question bank in `cookbooks/traffic/question_bank.json`.

This is a SAFETY-CRITICAL task:
- Missing a real collision is dangerous.
- False positives also hurt training quality.
- Classify based on **visible evidence** only. Do not speculate.

---

## What counts as a traffic accident here? 

You MUST choose the best matching category based on **temporal evidence across frames**:

- Decision priority (use this as a decision tree):
  1) If you can observe the accident process / transition during the segment → choose **B**.
  2) Else if you see any clear aftermath evidence in the segment → choose **C**.
  3) Else choose **A**.
  - If you are unsure between A vs C due to limited/ambiguous evidence, prefer **Unclear** (do not guess A).

- **A. No accident in the video**
  - Use **A only if** you can confidently verify across the sampled frames that there is **no accident evidence**:
    - no impact/contact sequence,
    - no fall process,
    - no fallen rider/pedestrian,
    - no debris field,
    - no abnormal post-collision vehicle positions.
  - Normal traffic flow or normal red-light stops/queues only.

- **B. The moment of collision is occurring in the video** (**collision happening now**)
  - You **observe the accident process** (the moment/sequence) within the segment, i.e., a **transition across frames** such as:
    - visible contact/impact happens in-frame, OR
    - a rider/pedestrian falls during the segment (standing/moving → falling → on ground), OR
    - a vehicle transitions into an abnormal stop/pose during the segment with collision cues (contact, debris appearing, immediate post-impact positioning).
  - Key requirement: describe **what changed across frames** that indicates the collision occurred during the video.

- **C. A collision occurred prior to the video** (**collision already occurred**)
  - You see **aftermath** evidence consistently in the frames (fallen motorcycle/person already on ground, debris already present, vehicles already stopped abnormally), but you do **NOT** observe the impact/fall/transition during this segment.
  - Key requirement: evidence is a **result state** with no observed collision sequence in the segment.

Boundary guidance:
- Normal red-light stops / neat queues alone ≠ collision.

---

## CRITICAL: Understanding your input

Videos may have **bounding boxes** overlaid on objects (cars, motorcycles, pedestrians). Look through the overlay and focus on the physical scene.

Small objects (motorcycles/bicycles/pedestrians) can be tiny. Actively scan:
- corners/edges,
- between large vehicles,
- crosswalk areas.

---

## Collision detection method (use this approach)

## IMPORTANT: Use multi-frame temporal reasoning (not single images)

You are given a set of frames sampled from a video segment. Your analysis MUST consider the **temporal relationship across frames**.

- Do NOT base conclusions on a single frame. Compare early/middle/late frames to detect **changes over time** (e.g., a person falling, a vehicle transitioning into an abnormal stop, debris appearing, bystanders gathering).
- When you claim an event is happening now vs. already happened, your evidence should reference a **transition across frames**, not just a static snapshot.
- Treat frames as a sparse sampling of a continuous video: use frame-to-frame differences to mentally reconstruct the dynamic progression of the scene.
- Do NOT invent motion/events that are not supported by differences across frames; if the progression cannot be established from the sampled frames, say Unclear.

### Step 1 — scan for definitive collision evidence
- Person lying flat on roadway/crosswalk
- Motorcycle/bicycle on its side (horizontal)
- Debris field (plastic/glass) on roadway
- Vehicles in impossible positions (sideways, diagonal, stopped in intersection center)
- Multiple vehicles clustered abnormally close (<1m) in a non-queue location

### Step 2 — distinguish normal stops from collision-related stops
Normal (NOT a collision by itself):
- Vehicles stopped at stop line due to red light
- Vehicles queued in a lane with consistent spacing

Collision-likely:
- Stopped in **intersection center**
- At odd angles, blocking lanes
- Bystanders clustered around a point
- One vehicle stopped next to fallen rider/motorcycle within ~1m–3m

### Critical false positives to avoid
- Heavy traffic volume / long queues alone ≠ collision.
- Vehicles stopped neatly behind stop line ≠ collision.
- A parked scooter/bicycle on roadside ≠ collision (needs abnormal on-road positioning).
- Do NOT infer accident just because "it looks like it could have happened" — require evidence.

### Step 3 — decide collision timing for this window
- If you see the **impact / fall / sudden stop signature** within this window → collision happening now.
- If you only see aftermath with no impact moment → collision already occurred.

### Vehicle involvement (presence ≠ involvement)
When you describe involvement for follow-up questions later:
- **Involved**: touching / within ~1m of the collision center, abnormal stop/angle, visible damage, directly interacting with debris/person.
- **Not involved**: passing through normally, stopped at stop line, parked curbside, >5m away with no interaction.
If uncertain, say **Unclear** (do not force involvement).

---

## Red light running (for later mapping)

Only claim red-light running if there is **clear visual evidence**:
- Traffic signal is red for that approach, AND
- A moving vehicle crosses the stop line / enters intersection against red.

If the signal is not visible or timing is unclear, state it as **unknown/unclear** (do not assert it happened).

---

## Weather / road / time cues (evidence-based)

Describe what you can directly observe.

- **Clear weather evidence**: blue sky / direct sun / sharp shadows / high visibility.
- **Cloudy/overcast evidence**: gray/white sky / diffused light / soft/no shadows.
- **Rainy evidence**: visible rain streaks OR droplets on lens OR active precipitation; wet road alone is not enough to claim rain.
- **Foggy evidence**: reduced visibility / haze where distant objects fade.
- **Roads wet**: visible reflective wet sheen, puddles, spray trails, or dark wet asphalt.
- **Time of day**:
  - Day: daylight scene.
  - Night: dark scene dominated by artificial lights/headlights.
  - Other: dawn/dusk/twilight, tunnels/indoor parking, or ambiguous lighting.

---

## Pedestrian definition (IMPORTANT)

In this project, **pedestrian** means a **person on foot** (standing or walking).

- A motorcycle/scooter/bicycle **rider** is **NOT** a pedestrian.
- A person **inside a vehicle** (driver/passenger) is **NOT** a pedestrian.
- A person **standing or walking** on sidewalk, island, crosswalk, or roadway **IS** a pedestrian.

## Pedestrians crossing

Only say pedestrians are crossing if a **pedestrian (person on foot)** is clearly in the roadway/crosswalk moving across lanes (not just standing on sidewalk/island).

---

## OUTPUT FORMAT (strict)

Output a **single structured block** exactly like this (fill values; keep headings):
The labels below are evidence cues for the downstream mapper, not final MCQ answers. Do not output MCQ JSON; the mapper will convert your evidence into exact option strings from `cookbooks/traffic/question_bank.json`.

```
[CATEGORY: ACCIDENT / NORMAL TRAFFIC]
[CONFIDENCE: high / medium / low]

[Accident Summary]:
- 1_01 (Is there a traffic accident in the video?): No accident in the video/The moment of collision is occurring in the video/A collision occurred prior to the video/Unclear. Evidence: ...

[Participants & Involvement]:
- 2_01 (Is a sedan involved in the accident?): Yes/No/Unclear. Evidence: ...
- 2_02 (What is the color of the sedan in the accident?): Black/White/Gray/Blue/Red/Green/Other/Unclear. Evidence: ...
- 2_03 (Is a motorcycle or bicycle involved in the accident?): Yes/No/Unclear. Evidence: ...
- 2_04 (Is a walking pedestrian involved in the accident?): Yes/No/Unclear. Evidence: ...

[Signals & Violations]:
- 2_05 (Is a vehicle running a red light?): Yes/No/Unclear. Evidence: ... (include whether the signal is visible)

[Emergency / Responders]:
- 1_02 (Are police vehicles present in the scene?): Yes/No/Unclear. Evidence: ...
- 1_08 (Are ambulances present in the scene?): Yes/No/Unclear. Evidence: ...
- 1_09 (Are fire trucks present in the scene?): Yes/No/Unclear. Evidence: ...

[Weather / Road / Time]:
- 1_03 (Is there visible evidence that the weather is clear in the scene?): Yes/No/Unclear. Evidence: ...
- 1_04 (Is there visible evidence that the weather is cloudy or overcast in the scene?): Yes/No/Unclear. Evidence: ...
- 1_05 (Is there visible evidence that it is raining in the scene?): Yes/No/Unclear. Evidence: ...
- 1_06 (Is there visible evidence that the weather is foggy in the scene?): Yes/No/Unclear. Evidence: ...
- 1_10 (Are the roads wet?): Yes/No/Unclear. Evidence: ...
- 1_07 (What is the time of day in the scene?): Day/Night/Other/Unclear. Evidence: ...

[Pedestrians]:
- 1_11 (Are there any pedestrians walking in the scene?): Yes/No/Unclear. Evidence: ...
- 2_06 (Are there any pedestrians walking across the road in the scene?): Yes/No/Unclear. Evidence: ...

[Road Layout & Traffic Pattern]:
- 3_01 (Describe the road layout and traffic pattern.): Brief description. Evidence: ...

[Extra Notes]:
- Briefly list any key objects/positions that support your judgments (debris, stopped angles, people on ground, etc.).
```

Hard rule: if something is not clearly visible, prefer **Unclear** rather than guessing.
