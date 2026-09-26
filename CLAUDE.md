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
- `server/app.py`: `POST /api/search` starts a simulated search and `GET /api/search/<id>` polls it. The progress fields are `stage`, `total_photos`, `photos_taken`, `photos_checked`, `distance_px`, and `failed_at`. A query containing `fail` simulates a miss. Search state is in memory, one active search at a time, and scoped to the signed-in patient.
- `server/camera.py` and `capture_frames.py`: camera stream utilities kept for future real object search. The current Flask app does not expose a camera preview endpoint.

## Team and hardware context

The camera is an iPhone running DroidCam over its hotspot, commonly at `http://172.20.10.1:4747/video`. The Arduino connects by USB; motors need separate battery power, and a transistor switches the laser. Person B is the hardware point of contact.

The teammate's vision module takes angle-tagged scan photos and a query, then returns object status, a bounding box, azimuth, and elevation. The proposed convention is azimuth 0 at pan home, positive right; elevation 0 horizontal, positive up. Confirm the final JSON contract and calibration before integration. `ambiguous` and `not_found` results must not activate the laser. The real scan, turret control, laser detection, and correction loop are not yet integrated here.

## Data precautions

- `.env` and generated captures are ignored. Never print, commit, or copy secret values into docs or tests.
- `server/data/auth.sqlite3*` is ignored local account data shared by checkouts of this working directory. Preserve it unless explicitly asked to remove it.
- Check `git status` before editing; preserve unrelated work. Verify code rather than treating this document as an exact specification when implementation changes.
