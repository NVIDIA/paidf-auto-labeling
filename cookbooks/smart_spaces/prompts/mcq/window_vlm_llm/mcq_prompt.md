# Smart Spaces Event Verification — Evidence to Answers

**Purpose:** Convert the structured evidence into a strict JSON object answering the question bank in `cookbooks/smart_spaces/question_bank.warehouse_event_verification_qa.json`.

**Synchronization note:** The decision rules below embed judgment calls (severity levels, PPE compliance thresholds, category tie-breaks) that aren't expressible in the question bank JSON itself; the question list is injected at `{question_bank_json}` below so it always mirrors the canonical bank.

---

## YOUR TASK

You will receive structured evidence (from a VLM) describing a clip from a monitored space. Your job is to output **ONLY** a JSON object with answers to the authoritative question bank below.

---

## OUTPUT FORMAT (strict)

Output **ONLY** a JSON object with this exact structure and no prose outside the JSON fence:

```json
{
  "items": [
    {
      "id": "aes_is_anomalous",
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
- For every question, `answer` MUST be **exactly equal** to ONE of the provided `options` strings, copied character-for-character.
  - Some options include a letter prefix (e.g. `"A. Yes"`) — include the prefix when the option has one.
  - Other options have no letter prefix (e.g. `"warehouse safety violation"`, `"moderate"`) — copy them exactly as given; do NOT invent a prefix.
- If a question has an empty `options` list (`[]`), you still MUST include a non-empty `answer` string.
  - The `answer` should be concise free text grounded in the evidence, not copied from an option.
  - Replace any placeholder text with the actual answer; do NOT output a placeholder literally.
- Yes/No questions: answer **"A. Yes" only with affirmative evidence**. If uncertain or not stated, prefer **"B. No"**.
- Every question in the bank must be answered — this bank has no `include_if` gating.
{reasoning_rule}

---

## QUESTION BANK (AUTHORITATIVE)

You MUST answer **every** question listed below, using the exact `id`, `question`, and `options`. Do NOT invent new questions or options.

```json
{question_bank_json}
```

---

## DETAILED DECISION RULES (more careful mapping)

If the evidence says "Unclear" for a detail, treat it as **insufficient evidence** and answer conservative negatives, except where a rule below says otherwise.

### Anomaly presence (aes_is_anomalous)
- "A. Yes" if the evidence describes anything abnormal, unsafe, dangerous, or otherwise noteworthy — including a safety violation, security issue, hazard, or person in distress — even if minor.
- "B. No" only if the evidence describes a routine, uneventful scene with no such detail.

### Primary anomaly category (aes_primary_category)
- Pick exactly one option: the category the evidence's `[ANOMALY SUMMARY]` names as the anomaly.
- If the evidence names more than one plausible category, choose the one most directly tied to the event's actual harm, risk, or intervention need (for example, prefer "physical fight" over "suspicious running" if a fight is what caused the running).
- Distinguish "robbery or mugging" (a confrontation or threat directed at a person) from "stealing or shoplifting" (quietly taking goods without confronting anyone) using the evidence's security-and-crime section.
- Distinguish "warehouse safety violation" (a PPE, lifting, or procedural violation in an industrial/warehouse setting with no other named category) from "industrial spill or leak" (a leaking or spilled substance is the primary issue).
- If the evidence's `[ANOMALY SUMMARY]` says no anomaly is visible, answer **"normal / no anomaly"**.

### Operational task (aes_operational_task)
- Pick the option matching the `[SETTING]` evidence's primary activity. Use "other" only if none of the listed tasks fit.

### PPE compliance (aes_ppe_compliance)
- "fully compliant": all visible people who need PPE are wearing the required PPE (e.g. high-visibility vest) with no gaps described.
- "partially compliant": at least one visible person has PPE and at least one visible person is missing required PPE, or PPE use is inconsistent.
- "non-compliant": no visible person has the required PPE despite the setting calling for it.
- "no people visible": the evidence reports no people in the clip.
- "not applicable": the setting is not one where PPE would normally apply (e.g. a residential or retail scene with no industrial task).

### Proper lifting (aes_proper_lifting)
- "A. Yes" only if the evidence explicitly describes bending at the knees during a lift or set-down.
- "B. No" if the evidence explicitly describes bending at the waist, or describes an awkward/unsafe carry.
- If no lifting or carrying is described, answer **"B. No"** (the question defaults negative when the technique cannot be confirmed as proper).

### Severity (aes_severity)
- "none": no anomaly is present.
- "low": a minor procedural or PPE gap with no injury, damage, or urgent risk described.
- "moderate": a clear violation, hazard, or incident with limited risk or impact (e.g. a near-miss, an isolated spill, a contained altercation).
- "high": a violation or incident with real injury, damage, or an active ongoing risk to people.
- "critical": severe, ongoing danger to life, a weapon/gunfire event, a large-scale hazard (fire, structural collapse, flooding), or clear serious injury.
- "unclear" evidence about severity should still be mapped to the closest supported level using the other evidence sections (injury, damage, intervention-required) rather than left unanswered.

### Ongoing at end (aes_ongoing_at_end)
- "A. Yes" only if the evidence explicitly states the event is still unfolding in the last frames.
- "B. No" if the evidence describes a resolved state, or if there is no anomaly.

### Temporal extent (aes_temporal_extent)
- Use the evidence's stated timing. If the event spans most of the clip, answer "throughout". If there is no anomaly, answer "not applicable".

### Actor count (aes_actor_count)
- Count only people the evidence describes as directly involved in the primary event, not bystanders. Use "none" if there is no anomaly or no person is involved.

### Free-text questions (aes_event_description, aes_operational_task_desc, aes_incident_category_desc, aes_precursors, aes_consequences, aes_search_query)
- Ground every sentence in the evidence. If the scene is normal, say so explicitly rather than inventing an incident.
- `aes_search_query` must be a single concise sentence, at most 20 words, written as a natural-language search query.
