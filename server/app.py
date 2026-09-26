"""Flask backend for the Theia UI.

Run with:  python server/app.py
"""
import itertools
import json
import random
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, g, jsonify, request, send_from_directory
from openai import APIStatusError

from auth import init_auth
from camera import CameraError, SharedCamera
from catalog import add_stock, find_location, load_catalog
from label_reader import LabelReaderError, read_label

app = Flask(__name__)
init_auth(app)

SCANS_DIR = Path(__file__).parent / "data" / "scans"
PREVIEW_WIDTH = 640
PREVIEW_FPS = 12

shared_camera = SharedCamera()

TOTAL_PHOTOS = 9
ON_TARGET_PX = 3

_ids = itertools.count(1)
_lock = threading.Lock()
_search = None


def _update(search_id, **changes):
    """Apply changes to the current search. Returns False if a newer search replaced it."""
    with _lock:
        if _search is None or _search["id"] != search_id:
            return False
        _search.update(changes)
        return True


def simulate_search(search_id, query):
    """Stand-in for the turret pipeline until the hardware and Grok steps are connected.

    Queries containing "fail" simulate Grok not finding the object in any photo.
    """
    time.sleep(0.5)
    for taken in range(1, TOTAL_PHOTOS + 1):
        if not _update(search_id, photos_taken=taken):
            return
        time.sleep(0.5)

    if not _update(search_id, stage="searching"):
        return
    time.sleep(0.4)
    for checked in range(1, TOTAL_PHOTOS + 1):
        if not _update(search_id, photos_checked=checked):
            return
        time.sleep(0.4)

    if "fail" in query.lower():
        _update(search_id, stage="failed", failed_at="searching")
        return

    distance = random.randint(60, 120)
    if not _update(search_id, stage="pointing", distance_px=distance):
        return
    while distance > ON_TARGET_PX:
        time.sleep(0.4)
        distance = int(distance * 0.6)
        if not _update(search_id, distance_px=distance):
            return
    time.sleep(0.4)

    _update(search_id, stage="on_target")


@app.post("/api/search")
def start_search():
    global _search
    text = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not text:
        return jsonify(error="query is required"), 400
    print(f"Query received: {text}")

    search = {
        "id": next(_ids),
        "user_id": g.user["id"],
        "query": text,
        "stage": "scanning",
        "total_photos": TOTAL_PHOTOS,
        "photos_taken": 0,
        "photos_checked": 0,
        "distance_px": None,
        "failed_at": None,
    }
    with _lock:
        _search = search
        snapshot = dict(search)
    threading.Thread(target=simulate_search, args=(search["id"], text), daemon=True).start()
    return jsonify(snapshot)


@app.get("/api/search/<int:search_id>")
def search_status(search_id):
    with _lock:
        if _search is None or _search["id"] != search_id or _search["user_id"] != g.user["id"]:
            return jsonify(error="search not found"), 404
        return jsonify(dict(_search))


@app.get("/api/catalog")
def catalog():
    return jsonify(load_catalog())


@app.get("/api/camera/preview")
def camera_preview():
    """Live MJPEG feed from the phone, for showing in an <img> tag."""
    try:
        shared_camera.read(timeout=5.0)
    except CameraError as error:
        return jsonify(error=str(error)), 503

    def frames():
        last_time = 0.0
        while True:
            try:
                frame, last_time = shared_camera.read(newer_than=last_time, timeout=5.0)
            except CameraError:
                return
            height, width = frame.shape[:2]
            if width > PREVIEW_WIDTH:
                frame = cv2.resize(frame, (PREVIEW_WIDTH, round(height * PREVIEW_WIDTH / width)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(1 / PREVIEW_FPS)

    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


def _scan_error_message(error):
    """Extract the provider's useful message without exposing its JSON wrapper."""
    if isinstance(error, APIStatusError):
        body = error.body
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = None
        details = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(details, dict):
            code = details.get("code")
            if code == "invalid_image_format":
                return "This photo format isn't supported. Choose a JPEG or PNG image."
            message = details.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if error.status_code == 429:
            return "The vision service is busy. Please try again shortly."
    return "Could not read the label right now. Please try again."


@app.post("/api/stock/scan")
def stock_scan():
    """Photograph the box held up to the phone camera and read its label.

    Sends the frame from the DroidCam stream by default. For testing without
    the camera, a photo can be uploaded instead as a multipart file field
    named "photo".
    """
    upload = request.files.get("photo")
    if upload is not None:
        image_bytes = upload.read()
        # The file picker can still supply HEIC, even with a JPEG/PNG accept hint.
        # Decode the actual bytes and store a real JPEG for both the model and preview.
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
        if frame is None:
            return jsonify(error="This photo format isn't supported. Choose a JPEG or PNG image."), 400
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return jsonify(error="Could not process this photo. Please choose another image."), 400
        image_bytes = encoded.tobytes()
    else:
        try:
            frame, _ = shared_camera.read(newer_than=time.monotonic(), timeout=10.0)
        except CameraError as error:
            return jsonify(error=str(error)), 503
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return jsonify(error="Could not encode the camera frame"), 500
        image_bytes = encoded.tobytes()

    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    scan_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (SCANS_DIR / f"{scan_id}.jpg").write_bytes(image_bytes)

    catalog_items = load_catalog()
    try:
        result = read_label(image_bytes, catalog_items)
    except LabelReaderError as error:
        return jsonify(error=str(error), scan_id=scan_id, image_url=f"/api/stock/scans/{scan_id}.jpg"), 500
    except Exception as error:  # network or API failures
        app.logger.warning("Label reading failed: %s", type(error).__name__)
        return jsonify(error=_scan_error_message(error), scan_id=scan_id,
                       image_url=f"/api/stock/scans/{scan_id}.jpg"), 502

    fields = result["fields"]
    shelf, bin_ = (None, None)
    if fields.get("drug_name") and fields.get("strength"):
        shelf, bin_ = find_location(catalog_items, fields["drug_name"], fields["strength"])

    return jsonify(
        scan_id=scan_id,
        image_url=f"/api/stock/scans/{scan_id}.jpg",
        fields=fields,
        warnings=result["warnings"],
        suggested_location={"shelf": shelf, "bin": bin_},
    )


@app.get("/api/stock/scans/<scan_id>.jpg")
def stock_scan_image(scan_id):
    if not re.fullmatch(r"[0-9_]+", scan_id):
        return jsonify(error="bad scan id"), 400
    return send_from_directory(SCANS_DIR, f"{scan_id}.jpg", mimetype="image/jpeg")


@app.post("/api/stock/confirm")
def stock_confirm():
    """Add the reviewed box to the catalog."""
    body = request.get_json(silent=True) or {}

    def text(key):
        value = body.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    drug_name, strength = text("drug_name"), text("strength")
    if not drug_name or not strength:
        return jsonify(error="drug_name and strength are required"), 400

    expiration_date = text("expiration_date")
    if expiration_date:
        try:
            date.fromisoformat(expiration_date)
        except ValueError:
            return jsonify(error="expiration_date must be YYYY-MM-DD"), 400

    try:
        count = int(body.get("count", 1))
    except (TypeError, ValueError):
        return jsonify(error="count must be a whole number"), 400
    if count < 1:
        return jsonify(error="count must be at least 1"), 400

    bin_ = body.get("bin")
    if isinstance(bin_, str):
        bin_ = int(bin_) if bin_.strip().isdigit() else (bin_.strip() or None)

    entry = add_stock(
        {
            "drug_name": drug_name,
            "strength": strength,
            "lot": text("lot"),
            "ndc": text("ndc"),
            "expiration_date": expiration_date,
            "box_color": text("box_color"),
            "shelf": text("shelf"),
            "bin": bin_,
        },
        count=count,
    )
    print(f"Stocked {count} x {drug_name} {strength} (lot {entry.get('lot')}) -> count {entry['count']}")
    return jsonify(entry)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
