"""Vision hook for the UI and the laser program.

Existing search, camera, and laser modules are unchanged. Call this when
JPEG frames from the camera are ready:

    from vision_hook import aim

    aim(frames, query)

frames is a list of {"image": jpeg_bytes, "pan": degrees, "tilt": degrees}
with an optional "timestamp" string. pan 0 is pan home and positive is right.
tilt 0 is horizontal and positive is up.

The return value is:

    {
      "fire_laser": false,
      "aim": null,
      "items": [
        {
          "id": "obj_001",
          "label": "lamp",
          "azimuth_deg": 30.0,
          "elevation_deg": -5.0,
          "confidence": 0.45
        }
      ]
    }

fire_laser is true only when the query status is found. aim is then
{"azimuth_deg", "elevation_deg"} and is null otherwise. Azimuth 0 is pan
home, positive is right. Elevation 0 is horizontal, positive is up. Degrees.
range_m stays null inside the vision package and is omitted here.

XAI_API_KEY is read only inside the vision process, from the environment or
a gitignored .env. A missing key raises MissingAPIKeyError with a short
message that points at .env. The key is never included in the return value.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vision.aim import aim

__all__ = ["aim"]
