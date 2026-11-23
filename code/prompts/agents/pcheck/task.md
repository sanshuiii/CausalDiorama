You check the object detections of one video before they are used. Do not use any tools other than reading the pictures listed below and the one write described at the end.

A detector looked for each entity of the list below on 12 frames of the video (frame numbers at 24 fps). It sometimes boxes the wrong
object (something similar that is not in the list, or the background), or boxes one listed entity under the name of another (two people swapped).

Entities of the video (id: short text -- appearance):
{ENT_LINES}

Detections to check (entity: frames with a box):
{DET_LINES}

Pictures:
{STRIPS}
{FRAMES}

For EVERY detection (entity x frame with a box) decide one verdict:
  "ok"            the box shows this entity (partly visible or cut by the frame edge is fine);
  "is:<other id>" the box shows another entity of the list instead (e.g. two people swapped);
  "drop"          the box shows something that is not this entity and not another listed entity (a similar object, the background, a
                  random part of the picture), or it is so loose / so wrong that it would mislead.
Compare people by their clothes, hair and position; compare objects by their look. Use the whole-frame pictures for positions.
When unsure, prefer "ok".

Write a JSON object {{"<entity id>": {{"<frame number>": "<verdict>", ...}}, ..., "notes": "<one or two sentences on what you changed and why>"}}
covering every detection listed above, to {OUT}, and reply only "done {N4}".
