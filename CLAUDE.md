# Theia — project context for coding assistants

## Current direction

Theia is a HackGT 13 object finder. A user asks where an item is; the future camera and turret pipeline will locate it and point a laser at it. The current item search is simulated. `main` is the patient experience only. The former clinic warehouse application, including staff accounts, box stocking, sample catalog, and inventory dashboard, is preserved on the separate local `warehouse` branch. Do not reintroduce those flows on `main` without an explicit request.

## Run and verify

- Install Python dependencies with `pip install -r requirements.txt`, then run `python server/app.py` at `http://127.0.0.1:5000`.
- In `ui/`, run `npm install` if needed and `npm run dev`; open `http://localhost:5173`. Vite proxies `/api` to Flask.
- Run backend tests with `python -m unittest discover -s server -p 'test_*.py'` from the repo root. Run `npm run build` and `npm run lint` from `ui/`.

## Current implementation

- `ui/src/AuthScreen.jsx` and `server/auth.py`: patient email/password registration, login, and logout. Passwords are hashed; revocable sessions use an HttpOnly cookie and SQLite at `server/data/auth.sqlite3`. The API rejects staff registration and login. Old staff records in a local database from the warehouse branch cannot authenticate on `main`.
- `ui/src/App.jsx`, `InputScreen.jsx`, and `ProgressScreen.jsx`: patient item query by typing or browser speech recognition, search progress, and spoken success/failure. There is no staff navigation or active catalog route.
- `server/app.py`: `POST /api/search` starts a search and `GET /api/search/<id>` polls it. The progress fields are `stage`, `total_photos`, `photos_taken`, `photos_checked`, `distance_px`, and `failed_at`. Search state is in memory, one active search at a time, and scoped to the signed-in patient.
- `server/object_search.py` and `server/scan.py`: a live search sweeps the turret through 7 pans, saves the angle-tagged photos under `server/data/scans/`, asks `vision/aim.py` where the object is, and on a confident `found` aims the laser and lights it for `LASER_ON_SECONDS` (default 15; 0 aims without lighting). Replaying a saved scan (`scan_dir`) never moves the turret.
- `server/turret.py` and `firmware/theia_turret/`: serial driver and Arduino firmware (115200 baud text lines: `PING`, `HOME`, `GOTO <pan> <tilt>` in servo degrees, `LASER 1|0`). Servo/vision conversion lives only in `turret.py`. Bench values: pan servo 0 is home and increasing servo angle turns left (`PAN_SIGN = -1`), so the sweep is vision pan 0 to -180; tilt servo 0 is level and 45 is down, so the laser reaches elevation 0 to -45 only. The laser is on pin D8, active-high. The firmware turns the laser off on every move and after 20 s. Configure with `TURRET` (`serial` or `none`), `ARDUINO_PORT` (COM4 on the bench laptop), `ARDUINO_BAUD`, and `SEARCH_SETTLE_S`. `TURRET=none` keeps the handheld scan with every photo at pan 0. Close the Arduino Serial Monitor before running the server.
- `server/camera.py` and `capture_frames.py`: DroidCam stream utilities.

## Team and hardware context

`server/laser.py` implements a red laser off/on detector with optional camera alignment and a pixel-space correction step. `laser_test.py` logs results without UI or motor integration. Run `python laser_test.py --off "captures/Laser Off.png" --on "captures/Laser On.png" --target 1410 2333` for the local example. The target is a manually estimated center of the white electrical outlet in the 3024x4032 laser-on image. Coordinates use x right/y down; correction is target minus dot; the default stop radius is 5 pixels. Recompute after each future motor nudge using fresh frames and a target in the same coordinate system. Pixel-to-motor calibration remains separate.

The camera is an iPhone running DroidCam over its hotspot, commonly at `http://172.20.10.1:4747/video`. The Arduino connects by USB; motors need separate battery power, and a transistor switches the laser. Person B is the hardware point of contact.

The teammate's vision module takes angle-tagged scan photos and a query, then returns object status, a bounding box, azimuth, and elevation. The proposed convention is azimuth 0 at pan home, positive right; elevation 0 horizontal, positive up. Confirm the final JSON contract and calibration before integration. `ambiguous` and `not_found` results must not activate the laser. The laser dot detection and correction loop are not yet integrated.

## Data precautions

- `.env` and generated captures are ignored. Never print, commit, or copy secret values into docs or tests.
- `server/data/auth.sqlite3*` is ignored local account data shared by checkouts of this working directory. Preserve it unless explicitly asked to remove it.
- Check `git status` before editing; preserve unrelated work. Verify code rather than treating this document as an exact specification when implementation changes.
