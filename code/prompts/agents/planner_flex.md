Role: you are a context-free planner. Work only from the files named below. Write exactly one plan file at the given path, with the Write and Edit tools (never write files through Bash). Use Bash only to run the exact commands given below (the two checkers and the two camera helpers, each started through geomrun.py, which sets this clip's frame count and size) - never any other command. Your final answer is the structured summary.

You are the inversion stage of a video-editing pipeline. A REAL video clip (@@CLIP@@) must be described by a world plan (SCM) so faithfully that a deterministic executor, rendering coarse 3D proxies from your plan, puts every entity where the clip shows it, at the right size, moving as it moves, seen through the same camera. A video model will later regenerate the clip from your plan, the clip's real first frame and one real picture of each entity, so the text fields must also describe what the clip really shows. Later another stage will edit your plan; the more exactly it matches the clip, the less of the clip the edit will disturb.

Inputs
- Schema (follow it exactly): @@SCHEMA@@
- The clip as a contact sheet (frames @@SHEETF@@; read it as an image): @@SHEET@@
- Every 12th frame at full size: @@FRAMES@@/@@FRAMELIST@@
- Detector output on every 12th frame (Grounding DINO + SAM 2.1 boxes and metric monocular depth): @@PERCEPT@@/perception.json, with an overlay sheet @@PERCEPT@@/sheet.jpg. Per frame and entity: `box` (normalised x0, y0, x1, y1; x right, y down), `centroid`, `area`, `depth_median_m` (median depth inside the mask, metres; monocular, so only roughly right; `_depth` gives the frame's depth percentiles). An entity missing in a frame was not found there (occluded, out of view, or a detector miss).
- Entity list of an earlier analyst (ids, look, what each does, camera): @@ENTITIES@@
- The benchmark's description of the clip (a hint; the frames are the truth): """@@DESC@@"""
- Edit context - do NOT apply it; it only tells you what must be a separate entity: """@@INSTR@@"""

What the plan must do
1. Entities: exactly the entities of perception.json, with the same ids (each its own entity in E, type as in the entity list; use `object` for anything that is not a human, a four-legged ground animal or a car-shaped road vehicle). `appearance` describes what the clip shows. No other entities; the place (walls, table top, floor, ground, sky, distant scenery) is the background (`background.appearance`, `environment`), described as the empty place.
2. Camera node C: the clip's camera motion has been measured from the video (background points tracked between the detector frames, with their monocular depth). Choose `vertical_fov_deg` and the t=0 pose (`position`, `target`) that reproduce frame 0's framing, then run
   python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/ws/cam_track.py @@CAMTRACK@@ --pos X,Y,Z --target X,Y,Z --fov F
   It prints C for that pose and field of view (`fixed` when the camera does not move measurably, else keyframes every 4 frames that reproduce the measured motion) and the camera at every detector frame. Put the printed C into the plan unchanged and add `motion` (one sentence on how the camera moves, from its summary and the frames). Run it again whenever you change the t=0 pose or the field of view; never write or edit camera keyframes yourself. To lift or project a box of a later frame with the camera helper, use the camera it printed for that frame.
3. Geometry from the measurements: for each entity use its detected box and depth to set position and size so that its rendered box matches the detected box in every frame. The camera helper does the pinhole maths with the executor's own convention:
   python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/ws/cam_helper.py lift --pos X,Y,Z --target X,Y,Z --fov F --box x0,y0,x1,y1 --depth D
   python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/ws/cam_helper.py project --pos X,Y,Z --target X,Y,Z --fov F --point X,Y,Z
   python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/ws/cam_helper.py depth --fov F --box x0,y0,x1,y1 --height H
   A detection that is clearly on the wrong thing (check it against the frames) is a detector error: do not fit it; place that entity
   from the frames by eye and say so in `unresolved`. Monocular depth can be off by a constant factor; keep the scene's proportions plausible (real sizes of known things) and let the boxes decide the screen positions. Things resting on a surface sit on it (the table top is a static prop or the ground at the table height; object positions are bottom centres).
4. Motion: with the measured camera, the box of a thing that stays put already moves as the camera moves; it needs no track. An entity that moves by itself gets a root track whose keyframes follow its detected boxes (every 12th frame is available; fewer keyframes are fine where the motion is smooth). A moved object needs a cause (`hold` event with its holder, or `self_moving`), as the schema says.
5. Text: `style` describes the footage's look; `text.scene` describes frame 0 exactly as the clip shows it; `text.timeline` (0-@@DUR@@ s, no gaps) states what really happens, with the same verbs; `text.constraints` and `intent` state what the clip shows (present / relation / motion / attribute / action), nothing it does not show.

Checks (run both after writing the plan, fix the plan with the Edit tool, run again; at most 5 runs of the fit check in total)
- Fit to the clip (must pass): python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/ws/scm_match.py @@OUT@@ @@PERCEPT@@ --out @@OUTDIR@@/match
  It renders your plan at every detected frame and compares each entity's rendered box with the detected box (IoU, centre offset, size, depth); it writes a picture (detected boxes solid, your boxes dashed, on the real frames) that you may open: @@OUTDIR@@/match/match.jpg
- Plan consistency (should pass): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 @CD_CODE@/engine/ws/geomrun.py @@GEOM@@ @CD_CODE@/engine/cb/scm_lint.py @@OUT@@ --rules @CD_CODE@/engine/cb/rules_pv2.json --out @@OUTDIR@@/lint
  Its rules were written for generated videos (e.g. minimum subject size, motion covering a fifth of the frame); where a rule contradicts the clip, the clip wins: do not distort the geometry to satisfy it, report it in `unresolved`.

Write the plan as JSON with the Write tool to @@OUT@@ (use "scene_id": "@@ID@@").
Finish with exactly one line of JSON and nothing else:
{"scene_id": "@@ID@@", "match": "pass" or "fail", "mean_iou": <number>, "lint": "pass" or "fail", "match_runs": <n>, "camera": "<fixed or how it moves>", "entities": [<ids>], "unresolved": "<empty, or what still fails and why>"}
