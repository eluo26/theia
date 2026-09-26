# Theia — project context for coding assistants

## Purpose and demo

Theia is a HackGT 13 prototype. A camera-equipped turret scans a doctor's sample closet, identifies a requested drug sample, and points a laser at it. The current patient-facing item search is simulated while the real turret and vision pipeline are integrated. A staff workflow scans a sample box label into a local catalog. Demo drug names are fictional.

The Impiricus angle is clinic sample inventory, expiration alerts, lookup events, and a mock “Request more” message. Do not use SMS or imply a live Impiricus API integration. Helping people with dementia find household objects and a guided walker are broader ideas, not the current hackathon build.

## Stack and running locally

- Python 3.13 and Flask: `python server/app.py` serves `http://127.0.0.1:5000`.
- React and Vite: from `ui/`, run `npm install` if needed, then `npm run dev`; open `http://localhost:5173`. Vite proxies `/api` to Flask.
- `npm run build` and `npm run lint` run from `ui/`. Backend auth tests: `python -m unittest discover -s server -p 'test_*.py'` from the repo root.
- Python dependencies are in `requirements.txt`.
- No React router: `#/` is item search; `#/stock` is staff stocking. `ui/src/App.jsx` handles route and account access.

## Current implementation

- `server/app.py`: Flask endpoints and a roughly 15-second simulated search. `POST /api/search` starts it; `GET /api/search/<id>` polls it. The progress contract uses `stage`, `total_photos`, `photos_taken`, `photos_checked`, `distance_px`, and `failed_at`. A query containing `fail` produces a simulated failure. Searches are currently in memory, with one active search at a time; a user cannot read another user's search status.
- `ui/src/InputScreen.jsx` and `ProgressScreen.jsx`: typed or browser speech query, simulated scanning/searching/pointing progress, and spoken result. Browser speech recognition works in supported browsers only.
- `server/camera.py`: one shared connection to the iPhone DroidCam stream, with fresh-frame reads and idle closing. `GET /api/camera/preview` provides MJPEG for staff stocking. `capture_frames.py` saves standalone captures.
- `server/label_reader.py`, `server/catalog.py`, and `ui/src/StockingScreen.jsx`: read a photographed or uploaded label with vision, let staff review/edit fields, then add stock. Handwritten mock labels are valid input. Catalog entries are per drug, strength, and lot. `GET /api/catalog`, `POST /api/stock/scan`, `GET /api/stock/scans/<id>.jpg`, and `POST /api/stock/confirm` support this flow.
- `server/auth.py`, `ui/src/AuthScreen.jsx`, and `ui/src/App.jsx`: local email/password registration, login, and logout. Roles are `patient` and `staff`; patients get item search, while staff also get stocking. Flask enforces staff access to stocking, catalog, and camera endpoints. Accounts and revocable sessions are stored in SQLite at `server/data/auth.sqlite3`; passwords are hashed and session IDs are held in an HttpOnly cookie. Staff registration is self-selected for the demo, so it is not a verified clinic identity system.

## Local data and secrets

- `server/data/catalog.json` is the demo catalog. Preserve entries and any additions made through stocking; do not casually reseed it.
- `.env` contains the real vision API key and is gitignored. Never print, commit, or copy its contents into docs or tests. `.env.example` documents configuration. `VISION_PROVIDER` is `openai` by default or `xai`; use `OPENAI_API_KEY` or `XAI_API_KEY`, with optional `VISION_MODEL` and `CAMERA_STREAM_URL`.
- `server/data/scans/`, `captures/`, and `server/data/auth.sqlite3*` are local generated data and gitignored. Preserve user data unless explicitly asked to remove it.
- The repo may contain unrelated uncommitted work. Inspect `git status` before editing and avoid overwriting it.

## Hardware and team integration

The camera is an iPhone running DroidCam over its hotspot, commonly `http://172.20.10.1:4747/video`. Keep DroidCam foregrounded; it may permit only one viewer. The Arduino connects to the laptop over USB; motors need separate battery power, and a transistor switches the laser. Person B is the hardware point of contact.

The teammate's vision module takes angle-tagged scan photos plus a query and returns object status, bounding box, azimuth, and elevation. The proposed convention is azimuth 0 at pan home, positive right; elevation 0 horizontal, positive up. Confirm the final JSON contract and calibration with the teammate before connecting it. `ambiguous` and `not_found` must not activate the laser. Real scan, turret control, laser detection, and correction loop are not yet integrated in this repo.

## Product decisions and remaining work

- Event log, mock Impiricus messages/endpoint, dashboard, and reorder UI remain to be built. The mock message should carry clinic, drug, time, and event type; show the outbound message when staff requests more stock.
- Shelf and bin exist in the catalog and stocking form, but their place in the product is undecided. The turret aims using camera angles, not stored shelf/bin coordinates.
- The plan suggests a technical view with scan images and alignment overlays, plus a simplified user view. The current UI is the simpler flow; overlays are future work.
- Treat this file as orientation, then verify behavior against the code before changing it. Update this file when major architecture or feature status changes.
