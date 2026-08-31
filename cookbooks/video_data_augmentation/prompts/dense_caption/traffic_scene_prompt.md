# Traffic Scene Prompt

You are given sampled frames from a traffic-camera video window. Produce one
factual dense caption grounded only in visible evidence.

Focus on road layout, camera angle, lanes/shoulders/medians, vehicles,
pedestrians/cyclists, stopped or disabled vehicles, abnormal poses, debris,
emergency/tow response, weather, lighting, road surface, and temporal changes
within this window. Mention signals, stop lines, crosswalks, lane numbers, or
responders only when they are actually visible.

Separate static aftermath from an observed active collision. If the evidence is
ambiguous, say what is visible and mark the event as uncertain rather than
choosing a verdict. Use visible track IDs only when an overlay label is readable
and attached to the described subject.

Do not infer driver intent, legal fault, license plates, identity, hidden causes,
or events outside the frames. Return concise prose suitable for downstream VQA
and reasoning.
