"""Flask backend for the Theia UI.

Run with:  python server/app.py
"""
import io
import itertools
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_file

from auth import init_auth
from object_search import public_search, run_object_search
from scan import ScanStoreError, resolve_saved_scan, scan_photo_count

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

app = Flask(__name__)
init_auth(app)

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


def _current_search(search_id):
    if _search is None or _search["id"] != search_id or _search["user_id"] != g.user["id"]:
        return None
    return _search


@app.post("/api/search")
def start_search():
    global _search
    body = request.get_json(silent=True) or {}
    text = (body.get("query") or "").strip()
    if not text:
        return jsonify(error="query is required"), 400
    replay_dir = None
    if body.get("scan_dir"):
        try:
            replay_dir = resolve_saved_scan(body.get("scan_dir"))
        except ScanStoreError as exc:
            return jsonify(error=str(exc)), 400
    print(f"Query received: {text}")

    search = {
        "id": next(_ids),
        "user_id": g.user["id"],
        "query": text,
        "stage": "scanning",
        "total_photos": scan_photo_count(),
        "photos_taken": 0,
        "photos_checked": 0,
        "progress_pct": 0,
        "detail": "Loading a saved scan…" if replay_dir is not None else "Connecting to the camera…",
        "saved_scan": None,
        "distance_px": None,
        "failed_at": None,
        "detection": None,
        "best_guess": False,
        "error": None,
        "preview_jpeg": None,
    }
    with _lock:
        _search = search
        snapshot = public_search(search)
    threading.Thread(
        target=run_object_search,
        args=(search["id"], text, _update),
        kwargs={"replay_dir": replay_dir},
        daemon=True,
    ).start()
    return jsonify(snapshot)


@app.get("/api/search/<int:search_id>")
def search_status(search_id):
    with _lock:
        search = _current_search(search_id)
        if search is None:
            return jsonify(error="search not found"), 404
        return jsonify(public_search(search))


@app.get("/api/search/<int:search_id>/preview")
def search_preview(search_id):
    with _lock:
        search = _current_search(search_id)
        if search is None:
            return jsonify(error="search not found"), 404
        preview = search.get("preview_jpeg")
        if not preview:
            return jsonify(error="preview not found"), 404
        data = bytes(preview)
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
