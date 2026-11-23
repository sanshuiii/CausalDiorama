# SCM schema `cb-scm-v1`, pipeline V2: camera node `C`, `hold` events, `self_moving`, edit declaration, each thing once

An SCM is one JSON object describing a 5.17-second, single-shot scene: 124 frames at 24 fps, and frame `f` is at time `t = f/24` seconds. A deterministic executor reads it and moves coarse 3D proxies (ellipsoids and boxes). It renders per-frame depth, semantic-class and instance maps. Those maps, the text fields, one reference picture per entity, one picture of the empty place and a first-frame image then condition a video generator. The executor reads only the fields listed here; put any reasoning into `notes`.

The proxies carry **only rough position, size, orientation and motion**. They never carry appearance, articulation, pose, facial expression or contact details; those come from the text. Plan geometry so that the viewer of the camera sees what the scene description asks for, and write the text so that it states the actions and appearances clearly.

## Coordinates and camera

- Units are metres, the coordinate system is right-handed, **Y is up**, and the ground is the plane `y = 0`.
- The camera is the SCM node **`C`** (below). It is fixed unless its controller moves it. It normally sits at negative Z and looks toward +Z. With that camera, **+X is screen right**, +Y is screen up, and larger Z is farther away.
- `C` (the camera node): `{"id": "camera", "controller": "fixed", "position": [x, y, z], "target": [x, y, z], "vertical_fov_deg": 35, "motion": "fixed camera"}`; `position`, `target` and `vertical_fov_deg` are the camera at t = 0. The output frame is 1344×768 (16:9).
- Framing rule of thumb: at `vertical_fov_deg = 35`, a subject 8 m in front of the camera and 1.8 m tall fills about 36 % of the frame height. Visible width at distance d is about `1.12 · d` metres; visible height is about `0.63 · d` metres.
- Typical sizes: an adult human is 1.6–1.9 m tall; a medium dog is 0.8 m long and 0.6 m tall; a cat is 0.45 × 0.3 m; a sedan is 4.6 × 1.45 × 1.8 m; a bicycle is 1.7 × 1.0 × 0.5 m; a small bird is 0.2 m.
- **Screen semantics.** Words such as left, right, above, below, in front of and behind in the scene description refer to **what the viewer sees** in this fixed camera:
  - left/right means screen x;
  - above/below means screen y;
  - in front of/behind means closer to or farther from the camera.
- **Camera node `C` and camera movement.** `C.controller` is one of:
  - `"fixed"`: the camera never moves.
  - `"keyframes"`: `"keyframes": [[t_seconds, {"position": [...], "target": [...], "vertical_fov_deg": 35}], ...]` (a keyframe may give only the fields that change; before the first keyframe the camera is the t = 0 pose, after the last one it holds), `"interp"`: `"smooth"` (default, eases in and out at every keyframe) or `"linear"`. Pan / tilt: move `target` only. Dolly in / out, truck, boom up / down: move `position` (and `target` with it to keep the angle). Zoom in / out: change `vertical_fov_deg` (smaller = closer).
  - `"orbit"`: `"orbit": {"sweep_deg": 30, "center": [x, y, z], "t0": 0, "t1": 5.17, "interp": "linear"}`: the camera swings around `center` (default: `target`) at constant distance and height; `sweep_deg` > 0 moves the camera to its right, < 0 to its left (arc shot).
  - `"follow"`: `"follow": {"entity": "<id>", "t0": 0, "t1": 5.17}`: position and target move with that entity (tracking shot).
  - Any controller may add `"look_at": "<entity id>"`: the target follows that entity, keeping its t = 0 offset.
  - `"motion"`: one sentence describing the camera in plain words, e.g. "The camera slowly pans to the right." or "Fixed camera."
  A different view without movement (high angle, low angle, side view, front view, wider or closer shot) is just a different `fixed` pose. Dependencies on the camera use paths such as `C.target` or `C.position` in `D`.

## Top-level fields

```jsonc
{
  "schema": "cb-scm-v1",
  "scene_id": "short_snake_case",
  "fps": 24,
  "frames": 124,
  "C": {...camera node...},
  "style": "one sentence: visual style and colour grading of the whole video",
  "background": {"appearance": "the empty place only"},
  "environment": {...},
  "E": [...entities...],
  "S": [...state tracks...],
  "R": [...shared events...],
  "D": [...dependencies...],
  "text": {...},
  "intent": {...},
  "edit": {...},                 // edited plans (B) only, see below
  "notes": "free text, not used by the executor"
}
```

### `environment`

```jsonc
{
  "ground": {"semantic": "road_paved" | "terrain" | "building_structure" | "water", "color": [r, g, b]},
  "sky": true,
  "backdrop": [
    {"id": "wall_1", "z": 18.0, "x_range": [-20, 20], "height": 9.0,
     "semantic": "building_structure" | "vegetation" | "infrastructure", "color": [r, g, b]}
  ],
  "static_props": [
    {"id": "hill", "center": [x, y, z], "radii": [rx, ry, rz],
     "semantic": "prop" | "vegetation" | "building_structure" | "infrastructure" | "terrain", "color": [r, g, b]}
  ],
  "description": "one paragraph: the look of the place, materials, lighting, weather, style"
}
```

- `backdrop` entries are vertical rectangles facing the camera at depth `z`. Each spans `x_range` horizontally and `0..height` vertically.
- `static_props` are ellipsoids that never move and have no appearance text. Use them only for scenery. **Every object named in the scene description must be an entity in `E`**, not a static prop.
- Colours are 0–1 floats and only affect a preview.
- `sky: true` fills everything not hit by geometry with sky. Use `false` for indoor scenes, and enclose the view with backdrops.

### `style` and `background` (reference pictures)

The video model gets one reference picture per entity (its look) and one picture of the empty place; they are drawn from these fields.
- `style`: one sentence for the visual style and colour grading of the whole video, e.g. `"photorealistic live-action footage, natural daylight colours"`, `"2D hand-drawn cartoon animation"`, `"oil painting with visible brush strokes"`, `"black-and-white film"`. Every reference picture and the first frame are drawn in this style.
- `background`: `{"appearance": "..."}`: the place alone, as a photograph of the empty scene would show it: ground, walls, sky, scenery, light, time of day, weather. **Each thing appears exactly once:** the place itself (a room, a cabin, a street, a field, a landscape) is the background and never also an entity; anything that is an entity of `E` must not be mentioned in the background text (the checker fails otherwise).

### `E`: entities

Every entity has:
- `id`: short snake_case;
- `type`;
- `instance_id`: a unique integer 1–250;
- `layer`: `"foreground"` for anything the scene description names, `"background"` for scenery that should act on its own, such as a waving flag or a fountain;
- `appearance`: concise, concrete visual description, including colours, species, material and clothing. **Put every attribute the scene description gives the entity here.**
- `label` (foreground entities): the plain common noun the scene description uses for it, one or two words, for example `"sheep"`, `"bicycle"`, `"ball"` or `"man"`. Several instances of the same thing share a label.
- `self_moving` (optional, `object` entities): `true` for a thing that moves on its own (a bird, a ball in flight, a drifting balloon, a boat, a drone, a disembodied hand). Any other object that moves must be moved by an entity through a `hold` event (below); the checker fails on objects that move by themselves.

| type | extra fields | notes |
|---|---|---|
| `human` | `height_m` | standing / walking figure, root on the ground |
| `animal` | `size_m`: `[length, height, width]` | four-legged ground animal, root on the ground |
| `vehicle` | `size_m`: `[length, height, width]` | car-shaped road vehicle only (car, van, bus, truck, tractor): the proxy is a car body on four wheels. Bicycles, motorcycles, scooters, carts, boats, aircraft and trains are `object` with `"semantic": "vehicle"`. |
| `object` | `size_m`: `[length, height, width]`, `shape`: `"ellipsoid"` or `"box"`, `semantic`, `position`: `[x, y, z]`, optional `heading_deg` | **anything else**: furniture, food, plants, balls, balloons, birds, fish, aircraft, boats, buildings, a tree, a cake… `position` is the **bottom centre**: `y = 0` rests on the ground, `y = 0.9` rests on a 0.9 m table, and `y = 3` hovers 3 m up. `semantic` is one of `prop`, `animal`, `vehicle`, `vegetation`, `building_structure`, `infrastructure`, `water`, `human`. Length runs along `heading_deg`, which defaults to 90, so the length is screen left-right. |
| `prop` | `size_m`: `[width, height, depth]` | small hand-held object passed in a `handoff` event only |
| `flag` | `pole_base`, `pole_height_m`, `flag_size_m`, `wave_period_s`, `wave_amplitude_m`, `direction_deg` | animates by itself |
| `fountain` | `center`, `basin_radius_m`, `jet_height_m` | animates by itself |

Articulated actions such as sitting, eating, climbing, swimming strokes, flapping or waving cannot be shown by the proxies. Give such an entity its rough position and motion, and state the action in the text.

### `S`: state tracks

Each moving entity needs exactly one root track. An entity without a track stays at its initial position.

```jsonc
{"id": "A_path", "entity": "A", "kind": "root_track", "interp": "linear" | "smooth",
 "keyframes": [[t_seconds, [x, z]], ...],       // human / animal / vehicle: on the ground
 "heading": "auto" | degrees}
{"id": "bird_path", "entity": "bird", "kind": "root_track", "interp": "smooth",
 "keyframes": [[t_seconds, [x, y, z]], ...],    // object: 3D, y = bottom height above ground
 "heading": "auto" | degrees}
```

- Before the first keyframe the entity stays at the first position; after the last keyframe it stays at the last one.
- `linear` gives constant speed. `smooth` eases in and out of every keyframe, which is how you express starting and stopping.
- `heading`:
  - `0` faces +Z (away from the camera), `90` faces +X (screen right), `-90` faces screen left, `180` faces the camera.
  - `"auto"` faces the direction of horizontal motion.
- To have a subject enter or leave the shot, start or end its track outside the camera's view.

### `R`: shared events

Two executable events:
- `handoff`: a hand-to-hand transfer of a `prop` between two humans: `{"id", "type": "handoff", "object", "giver", "receiver", "reach_start", "contact", "release", "settle", "meet_point"}`, with the times strictly increasing. Both humans must stand within about 0.7 m of `meet_point`.
- `hold`: an entity carries or moves an `object` / `prop`: `{"id", "type": "hold", "object": "<id>", "holder": "<id>", "start": t0, "end": t1, "mode": "carry" | "guide"}`.
  - `carry`: from `start` the object moves with the holder (a person's right-hand hold point in front of the chest, otherwise the holder's position), keeping its offset at `start`; after `end` it stays where it was released. Use it for picking up, holding, carrying, lifting.
  - `guide`: the object follows its own root track and a person holder's right hand stays on it. Use it for wiping, pushing, sliding, stirring, pouring motions where the hand moves the object along a path.
  - The holder must touch the object when the hold starts: a person must be within arm's reach (the executor fails otherwise); any other holder must be in contact (the checker measures it). Move the holder to the object first, then start the hold.
  - Whenever a subject picks up, holds, carries, hands over, wipes, pushes, pours or otherwise moves a thing, write the matching event: nothing moves without a cause.
Leave `R` empty when nothing is handed over or held.

### `edit`: the edit declaration (edited plans only)

An edited plan B declares exactly what it changed relative to the source plan A; the checker compares A and B and fails on any undeclared change.

```jsonc
"edit": {
  "look_changed": ["<ids whose look changes: colour, material, identity, clothing, shape details>"],
  "geometry_changed": ["<ids whose size, position, motion or holds change>"],
  "added": ["<new ids>"], "removed": ["<deleted ids>"],
  "camera_changed": false, "style_changed": false, "background_changed": false,
  "summary": "one sentence"
}
```

- `appearance` describes only how a thing looks. A size, position, count or motion edit changes geometry only and leaves `appearance` untouched; the thing's reference picture is then reused unchanged.
- Adding instances never changes the existing ones: keep their size, position, motion and look exactly as in plan A and put the new ones in free space.

### `D`: dependencies

`D` is a list of `{"from": "<path>", "to": "<path>", "why": "..."}` that records which fields must change together. Paths start with `E`, `S`, `R`, `C`, `environment`, `background`, `text`, `intent` or `style`, and name existing ids (wildcards such as `E.tomato_*` are allowed). The executor does not read it; the checker and the reviewer check that every edge is consistent.

### `text`

```jsonc
{
  "scene": "one paragraph establishing the setting, the camera and what is visible at t = 0 (who/what is where, with their appearance)",
  "timeline": [[t_start, t_end, "what happens in this interval, naming entities by their appearance"]],
  "constraints": ["short hard rules, e.g. exact counts, what must not happen"]
}
```

- The timeline must cover 0 to 5.17 s without gaps.
- Every action in the scene description belongs in the timeline, stated with the same verb.
- State counts explicitly, for example "exactly five bears".

### `intent`: what the video must visibly show

The executor does not read this block. A checker renders your plan and verifies each requirement, so write down **every requirement you can read from the scene description**, in screen terms. Each requirement is one object:

| kind | fields | what the checker verifies on the rendered proxies |
|---|---|---|
| `present` | `entities`, optional `interval` | each entity is visible and at least 8 % of the frame height in ≥ 90 % of the interval |
| `count` | `noun`, `n`, `entities` (exactly `n` ids), optional `interval` | exactly `n` of them are visible (≥ 5 % of the frame height) in ≥ 90 % of the interval |
| `relation` | `a`, `rel`, `b`, optional `interval` | `rel` ∈ `left_of`, `right_of`, `above`, `below`, `in_front_of`, `behind`, `on_top_of`, `under`. Holds on screen in ≥ 90 % of the frames where both are visible: centres at least 5 % of the frame apart, or at least 0.5 m apart in depth. `on_top_of` / `under` also need horizontal overlap. |
| `motion` | `entity`, `direction` ∈ `left`, `right`, `up`, `down`, `toward_camera`, `away_from_camera`, optional `interval` | net on-screen displacement of at least 20 % of the frame in that direction, moving mostly monotonically |
| `attribute` | `entity`, `phrase` | the phrase's words appear in that entity's `appearance` |
| `action` | `entity`, `phrase`, optional `interval` | a timeline entry in the interval describes it |
| `contact` | `a`, `b` | declares that a and b touch or overlap on purpose, for example a rider on a horse; relaxes the occlusion check |

- `interval` defaults to the whole shot, `[0, 5.17]`.
- The checker also requires every named entity to be at most 25 % hidden behind other things (60 % for entities in `contact`, `on_top_of` or `under` requirements).
- Also give `"expressiveness"`:
  - `"full"`: the proxies can show everything the scene description asks for;
  - `"partial"`: some of it is carried only by text;
  - `"text_only"`: the proxies add essentially nothing.

  Give `"expressiveness_reason"` in one sentence.

```jsonc
"intent": {
  "expressiveness": "partial",
  "expressiveness_reason": "Positions and the rising balloon are in the proxies; the dog's running gait is text only.",
  "requirements": [
    {"id": "r1", "kind": "present", "entities": ["dog", "bicycle"]},
    {"id": "r2", "kind": "relation", "a": "dog", "rel": "left_of", "b": "bicycle"},
    {"id": "r3", "kind": "action", "entity": "dog", "phrase": "running"},
    {"id": "r4", "kind": "attribute", "entity": "bicycle", "phrase": "red"}
  ]
}
```
