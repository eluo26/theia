"""Flask backend for the Theia UI.

Run with:  python server/app.py
"""
import itertools
import random
import threading
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

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
        if _search is None or _search["id"] != search_id:
            return jsonify(error="search not found"), 404
        return jsonify(dict(_search))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
