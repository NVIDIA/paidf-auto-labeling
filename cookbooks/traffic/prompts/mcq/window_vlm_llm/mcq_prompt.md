# MCQ Generation from Traffic Scene Description

**Purpose:** Convert the structured scene description into a strict MCQ JSON for the `cookbooks/traffic/question_bank.json` question bank.

**Synchronization note:** The embedded traffic question bank and decision rules below are intentionally duplicated from `cookbooks/traffic/question_bank.json`; keep question ids, text, options, `include_if`, and `aggregation` policy synchronized with that canonical source. This mode keeps a separate default prompt file so `window-vlm-llm` can evolve independently if its VLM-caption + LLM-mapping instructions diverge.

---

## YOUR TASK

You will receive a structured scene description (from a VLM). Your job is to output **ONLY** a JSON object with answers to the authoritative question bank below.

---

## OUTPUT FORMAT (strict)

Output **ONLY** a JSON object with this exact structure:

```json
{
  "version": 2.0,
  "video_id": "will_be_set_by_system",
  "mcq": [
    {
      "id": "1_01",
      "question": "…",
      "options": ["…"],
      "answer": "…"
    }
  ]
}
```

---

## GLOBAL RULES (MOST IMPORTANT)

- Output JSON only (no extra text).
- For every question, `answer` MUST be **exactly equal** to ONE of the provided `options` strings.
  - Copy/paste the option string character-for-character.
  - If an option includes a letter prefix (e.g. `"A. Day"`), you MUST include the prefix.
- If a question has an empty `options` list (`[]`), you still MUST include a non-empty `answer` string for that question.
  - The `answer` should be concise free text, not copied from an option.
  - Schema pattern: `{ "id": "3_01", "question": "Describe the road layout and traffic pattern.", "options": [], "answer": "<free-form description>" }`
  - Replace `<free-form description>` with the actual answer; do NOT output the placeholder literally.
- Yes/No questions:
  - Answer **"A. Yes" only with affirmative visual evidence** in the description.
  - If uncertain or not stated, prefer **"B. No"**.
- For `1_01` (traffic accident 3-way):
  - Choose exactly one of:
    - **"A. No accident in the video"** only if the description provides no accident evidence across the segment (no impact/contact sequence, no fall process, no debris, no abnormal post-collision vehicles).
    - **"B. The moment of collision is occurring in the video"** only if the accident/impact/fall transition is described as occurring during the video.
    - **"C. A collision occurred prior to the video"** if any clear aftermath evidence is described (fallen motorcycle/person on ground/debris/abnormal stopped vehicles) without the impact moment being described.
  - If the description is uncertain/ambiguous but hints at accident/aftermath, do NOT choose "A. No accident in the video"; prefer "C. A collision occurred prior to the video".
- `include_if`:
  - If a question has `include_if`, include it ONLY if the condition is satisfied based on YOUR OWN answers.
  - If not satisfied, OMIT the question entirely.

---

## QUESTION BANK (AUTHORITATIVE)

You MUST answer **ONLY** the questions listed below (respecting `include_if`).
Use the exact `id`, `question`, and `options`. Do NOT invent new questions or options.

```json
{
  "name": "traffic",
  "questions": [
    {
      "id": "1_01",
      "question": "Is there a traffic accident in the video?",
      "options": ["A. No accident in the video", "B. The moment of collision is occurring in the video", "C. A collision occurred prior to the video"],
      "aggregation": "majority"
    },
    {
      "id": "1_02",
      "question": "Are police vehicles present in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_03",
      "question": "Is there visible evidence that the weather is clear in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_04",
      "question": "Is there visible evidence that the weather is cloudy or overcast in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_05",
      "question": "Is there visible evidence that it is raining in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_06",
      "question": "Is there visible evidence that the weather is foggy in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_07",
      "question": "What is the time of day in the scene?",
      "options": ["A. Day", "B. Night", "C. Other"],
      "aggregation": "majority"
    },
    {
      "id": "1_08",
      "question": "Are ambulances present in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_09",
      "question": "Are fire trucks present in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_10",
      "question": "Are the roads wet?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "1_11",
      "question": "Are there any pedestrians walking in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "2_01",
      "question": "Is a sedan involved in the accident?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "2_02",
      "question": "What is the color of the sedan in the accident?",
      "options": ["A. Black", "B. White", "C. Gray", "D. Blue", "E. Red", "F. Green", "G. Other"],
      "aggregation": "majority",
      "include_if": { "2_01": "A. Yes" }
    },
    {
      "id": "2_03",
      "question": "Is a motorcycle or bicycle involved in the accident?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "2_04",
      "question": "Is a walking pedestrian involved in the accident?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "2_05",
      "question": "Is a vehicle running a red light?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any"
    },
    {
      "id": "2_06",
      "question": "Are there any pedestrians walking across the road in the scene?",
      "options": ["A. Yes", "B. No"],
      "aggregation": "any",
      "include_if": { "1_11": "A. Yes" }
    },
    {
      "id": "3_01",
      "question": "Describe the road layout and traffic pattern.",
      "options": [],
      "aggregation": "first"
    }
  ]
}
```

---

## CONSISTENCY RULES (apply to your own answers)

- Sedan involvement vs presence:
  - A sedan is "involved" only if the description indicates it is part of the accident (contact, abnormal stop at collision center, adjacent to fallen rider/person, damage/debris association).
  - If `2_01` = "B. No" → OMIT `2_02`.

- Red-light running:
  - Answer "A. Yes" only if the description explicitly provides evidence of a vehicle entering against a red signal.
  - If signal state is not described/visible → answer "B. No".

---

## DETAILED DECISION RULES (more careful mapping)

Use these rules to convert the structured description into answers. If the description uses "Unclear/unknown", treat it as **insufficient evidence** and answer conservative negatives (except `1_07` which allows "C. Other").

### Traffic accident type (1_01)
- Decision priority (use this as a decision tree):
  1) If the description indicates you can observe the accident process within the segment (a transition across frames: impact/contact, a rider/pedestrian falling during the segment, debris appearing, or a vehicle transitioning into an abnormal post-collision stop/pose) → **B**.
  2) Else if the description indicates any clear aftermath evidence (fallen motorcycle/person already on ground, debris present, abnormal stopped vehicles) → **C**.
  3) Else → **A**.
- **Never choose A** if the description is uncertain/ambiguous but mentions potential accident/aftermath cues; in that case choose **C**.

### Responders (1_02, 1_08, 1_09)
- Answer "A. Yes" only if explicitly described/visible.
- Do NOT assume responders are present just because there is an accident.

### Weather evidence (1_03–1_06)
These are **evidence questions**, not forced classification. "Yes" requires affirmative evidence.
- **1_03 Clear**: blue sky, direct sun, sharp shadows, high visibility.
- **1_04 Cloudy/overcast**: gray/white sky, diffused light, soft/no shadows.
- **1_05 Raining**: visible rain streaks/droplets OR active rainfall described. Wet road alone is not enough.
- **1_06 Foggy**: reduced visibility/haze described (distant objects fade).
If not mentioned/unclear → answer "B. No".

### Time of day (1_07)
- Choose exactly one:
  - **A. Day**: daylight / sunlit scene.
  - **B. Night**: dark scene, artificial lights/headlights dominate.
  - **C. Other**: dawn/dusk/twilight, indoor/tunnel, ambiguous.

### Roads wet (1_10)
- "A. Yes" only if wet sheen, puddles, spray trails, or explicit statement of wet road is in the description.
- If only "rainy evidence = B. No" but road described wet, still answer **1_10 = "A. Yes"** (wetness can remain after rain).

### Pedestrians crossing (2_06)
- Pedestrian definition: **pedestrian means a person on foot** (standing or walking).
  - A motorcycle/scooter/bicycle rider is NOT a pedestrian.
  - A person inside a vehicle (driver/passenger) is NOT a pedestrian.
- "A. Yes" only if a **pedestrian (person on foot)** is described as moving across the roadway/crosswalk (not standing on sidewalk/island).

### Pedestrians in the scene (1_11)
- Pedestrian definition: **pedestrian means a person on foot** (standing or walking).
  - A motorcycle/scooter/bicycle rider is NOT a pedestrian.
  - A person inside a vehicle (driver/passenger) is NOT a pedestrian.
- "A. Yes" if any **pedestrian (person on foot)** is visible anywhere in the scene (sidewalk, island, crosswalk, roadway), even if they are standing still. Otherwise "B. No".

### Accident participants (2_01–2_05)

#### Sedan involved (2_01) + sedan color (2_02)
- **2_01 = "A. Yes"** only if a sedan is described as part of the accident (contact, abnormal stop at collision center, adjacent to fallen person/two-wheeler, damage/debris association).
- **2_01 = "B. No"** if sedan is only passing by or stopped normally at a stop line.
- **2_02**: pick a color only if the sedan involved is described with a clear color cue; otherwise choose **"G. Other"**.

#### Motorcycle/bicycle involved (2_03)
- "A. Yes" if a motorcycle/scooter/bicycle is described as fallen, struck, or part of the crash. Otherwise "B. No".

#### Pedestrian involved (2_04)
- Pedestrian definition: **pedestrian means a person on foot** (standing or walking).
  - A motorcycle/scooter/bicycle rider is NOT a pedestrian.
  - A person inside a vehicle (driver/passenger) is NOT a pedestrian.
- "A. Yes" only if a **pedestrian (person on foot)** is described as struck, on ground due to crash, or otherwise part of the accident.
- If the person on the ground is a **rider** (e.g., next to a fallen motorcycle/scooter/bicycle), that is NOT a walking pedestrian → answer "B. No" for `2_04`.
- If pedestrians are merely crossing normally, that is NOT accident involvement.

#### Vehicle running red light (2_05)
- "A. Yes" only with explicit evidence a vehicle entered against a red signal.
- If signal visibility is unclear → "B. No".
