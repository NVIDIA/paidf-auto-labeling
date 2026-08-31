# Smart Spaces Scene Prompt

You are given sampled frames from a monitored-space video window — indoor or
outdoor, a road or vehicle area, a warehouse or industrial facility, a retail
or commercial space, a residential area, or another public or operational
setting. Produce one factual dense caption grounded only in visible evidence.

First note the setting (indoor/outdoor, road/vehicle, warehouse/industrial,
retail/commercial, residential, or other) and the primary activity underway,
then describe whatever is visible from these anomaly-relevant categories.
Only describe categories that actually apply to what is visible — do not
force every category into the caption:

- Traffic: vehicle or pedestrian collisions, near-misses, traffic-signal or
  right-of-way violations, and obstructions blocking a road, lane, or path.
- Animals: an animal on a road or in a work, storage, or restricted area, and
  any dangerous interaction between an animal and people or vehicles.
- Security and crime: unauthorized or forced entry; a person taking goods
  quietly (shoplifting or theft) versus a direct confrontation, threat, or
  taking-by-force against a person (robbery or mugging); physical
  altercations or fights; assault; a visible weapon, gunfire, muzzle flash,
  or people suddenly ducking, fleeing, or dispersing; vandalism; or a crowd
  turning into a large disruptive or violent disturbance.
- Fire, weather, and structural hazards: smoke, fire, or explosion; flooding
  or water intrusion; severe wind or storm damage; ground movement such as an
  avalanche or landslide; and structural collapse or debris.
- Person state and suspicious behavior: a person falling, collapsing, or
  otherwise in visible distress; a person running or moving in a way that is
  abnormal for the setting; and dangerous or suspicious objects left
  unattended.
- Falling or leaking hazards, in any setting: an object, debris, or material
  falling from height or from storage; and any liquid, chemical, or material
  leaking or spilling from a container, pipe, vehicle, or piece of equipment
  — not limited to a warehouse or a forklift.
- Industrial and warehouse operations (when the setting is warehouse or
  industrial): the operational task underway (walking/transiting, manual
  lifting or carrying, operating a forklift or machinery, stacking/shelving/
  retrieving goods, loading/unloading, cleaning/maintenance, inspecting/
  monitoring, or idle/standing); personal protective equipment such as
  high-visibility vests or hard hats present or absent on each visible
  person; lifting technique (bending at the knees versus the waist); and
  boxes, pallets, or items precariously placed, falling, or already fallen
  from a rack or shelf.
- Near-misses with equipment or vehicles: a person coming dangerously close
  to, or narrowly avoiding contact with, a forklift, other powered
  equipment, or a vehicle, in any setting.
- Access and area boundaries: people entering restricted, marked, or
  unattended areas, or interacting with equipment or containers they do not
  appear authorized to use.
- General hazards: spills, debris, or obstructions in a path that a person
  could trip over or collide with.

Separate a routine, uneventful scene from an observed unsafe act, violation,
or incident. If the evidence is ambiguous, say what is visible and mark the
detail as uncertain rather than asserting a violation, a crime, or an injury.
Use visible track IDs only when an overlay label is readable and attached to
the described subject.

Do not infer intent, identity, employment or legal status, or causes and
outcomes outside the frames. Return concise prose suitable for downstream
event-verification QA and anomaly reasoning.
