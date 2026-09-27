"""Run with python -m unittest discover -s server -p 'test_*.py'."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from object_search import announcement, annotate_detection, public_search, run_object_search
from scan import (
    ScanCancelled,
    ScanStoreError,
    capture_scan,
    encode_jpeg,
    load_saved_scan,
    resolve_saved_scan,
    save_scan,
    scan_photo_count,
)
from turret import FakeTurret, TurretError


def tiny_jpeg(width=40, height=30, color=(10, 20, 30)):
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return encode_jpeg(frame)


class FakeCamera:
    def __init__(self, width=40, height=30):
        self.width = width
        self.height = height
        self.t = 1.0
        self.reads = 0

    def read(self, newer_than=0.0, timeout=5.0):
        self.reads += 1
        self.t = max(self.t, newer_than) + 0.001
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame, self.t


class ScanTests(unittest.TestCase):
    def test_default_scan_matches_turret_arc(self):
        self.assertEqual(scan_photo_count(), 7)
        self.assertEqual(scan_photo_count(3.5, 0.5), 7)

    def test_handheld_scan_returns_jpegs_with_zero_angles(self):
        frames = capture_scan(camera=FakeCamera(), duration_s=0.15, interval_s=0.05)
        self.assertEqual(len(frames), 3)
        self.assertTrue(frames[0]["image"].startswith(b"\xff\xd8"))
        self.assertEqual(frames[0]["pan"], 0.0)
        self.assertEqual(frames[0]["tilt"], 0.0)
        self.assertIsInstance(frames[0]["timestamp"], str)

    def test_turret_scan_tags_real_pans(self):
        turret = FakeTurret()
        with patch.dict("os.environ", {"SEARCH_SETTLE_S": "0"}):
            frames = capture_scan(camera=FakeCamera(), turret=turret)
        self.assertEqual([frame["pan"] for frame in frames], [0, -30, -60, -90, -120, -150, -180])
        self.assertTrue(all(frame["tilt"] == 0.0 for frame in frames))
        self.assertEqual(turret.calls[:3], [("laser_off",), ("home",), ("goto", 0.0, 0.0)])

    def test_turret_scan_stops_when_replaced(self):
        turret = FakeTurret()
        checks = iter([True, True, False])
        with patch.dict("os.environ", {"SEARCH_SETTLE_S": "0"}):
            with self.assertRaises(ScanCancelled):
                capture_scan(camera=FakeCamera(), turret=turret, should_continue=lambda: next(checks))
        self.assertEqual(len([call for call in turret.calls if call[0] == "goto"]), 2)

    def test_save_scan_roundtrip_keeps_jpeg_and_angles(self):
        jpeg = tiny_jpeg()
        with tempfile.TemporaryDirectory() as directory:
            saved = save_scan(
                [{"image": jpeg, "pan": 30.0, "tilt": -5.0, "timestamp": "t"}],
                root=directory,
                query="blue bottle",
            )
            loaded = load_saved_scan(saved)
            manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((saved / "frame_000.jpg").is_file())
        self.assertEqual(loaded[0]["image"], jpeg)
        self.assertEqual(loaded[0]["pan"], 30.0)
        self.assertEqual(loaded[0]["tilt"], -5.0)
        self.assertEqual(loaded[0]["timestamp"], "t")
        self.assertEqual(manifest["query"], "blue bottle")

    def test_resolve_saved_scan_rejects_paths_outside_the_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "scan"
            inside.mkdir()
            (inside / "manifest.json").write_text("{}", encoding="utf-8")
            outside = Path(directory).parent / "elsewhere"
            with patch.dict("os.environ", {"SEARCH_SCAN_DIR": str(root)}):
                self.assertEqual(resolve_saved_scan("scan"), inside.resolve())
                self.assertEqual(resolve_saved_scan(str(inside)), inside.resolve())
                with self.assertRaises(ScanStoreError):
                    resolve_saved_scan("../elsewhere")
                with self.assertRaises(ScanStoreError):
                    resolve_saved_scan(str(outside))


class ObjectSearchTests(unittest.TestCase):
    def test_found_object_keeps_box_and_preview(self):
        jpeg = tiny_jpeg()
        search = {"id": 1}
        progress_events = []

        def capture(on_frame=None):
            frames = [{"image": jpeg, "pan": 0.0, "tilt": 0.0, "timestamp": "t"}]
            if on_frame:
                on_frame(1, 1, frames[0], None)
            return frames

        def locate(frames, query, on_progress=None):
            self.assertEqual(query, "mug")
            self.assertEqual(len(frames), 1)
            if on_progress:
                on_progress({"phase": "indexing", "done": 1, "total": 1})
                on_progress({"phase": "matching", "done": 1, "total": 1})
                progress_events.extend(["indexing", "matching"])
            return {
                "fire_laser": True,
                "aim": {"azimuth_deg": 4.0, "elevation_deg": -1.5},
                "result": {
                    "status": "found",
                    "label": "mug",
                    "bbox_px": [8, 6, 28, 24],
                    "center_px": [18, 15],
                    "image_width": 40,
                    "image_height": 30,
                    "confidence": 0.9,
                    "reason": "Matched the catalog label.",
                    "location": "next to the white energy drink can",
                },
                "frame_jpeg": jpeg,
            }

        def update(_search_id, **changes):
            search.update(changes)
            return True

        run_object_search(1, "mug", update, capture=capture, locate=locate)
        self.assertEqual(search["stage"], "on_target")
        self.assertEqual(
            search["detection"]["announcement"], "The mug is next to the white energy drink can."
        )
        self.assertEqual(search["detection"]["bbox_px"], [8, 6, 28, 24])
        self.assertEqual(search["detection"]["center_px"], [18, 15])
        self.assertEqual(search["detection"]["azimuth_deg"], 4.0)
        self.assertGreater(search["distance_px"], 0)
        self.assertEqual(search["progress_pct"], 100)
        self.assertEqual(progress_events, ["indexing", "matching"])
        self.assertTrue(search["preview_jpeg"].startswith(b"\xff\xd8"))
        public = public_search(search)
        self.assertNotIn("preview_jpeg", public)
        self.assertTrue(public["has_preview"])

    def test_camera_capture_is_saved_for_replay(self):
        jpeg = tiny_jpeg()
        search = {"id": 1}

        def capture(on_frame=None):
            frames = [{"image": jpeg, "pan": 30.0, "tilt": 0.0, "timestamp": "t"}]
            if on_frame:
                on_frame(1, 1, frames[0], None)
            return frames

        def locate(frames, query, on_progress=None):
            return {"fire_laser": False, "result": {"status": "not_found", "reason": "no"}}

        details = []

        def update(_search_id, **changes):
            search.update(changes)
            if changes.get("detail"):
                details.append(changes["detail"])
            return True

        with tempfile.TemporaryDirectory() as directory:
            run_object_search(1, "blue bottle", update, capture=capture, locate=locate, save_dir=directory)
            saved = Path(search["saved_scan"])
            loaded = load_saved_scan(saved)
            self.assertEqual(loaded[0]["image"], jpeg)
            self.assertEqual(loaded[0]["pan"], 30.0)
            replayed = {}

            def replay_capture(on_frame=None):
                raise AssertionError("replay should not open the camera")

            def replay_locate(frames, query, on_progress=None):
                replayed["frames"] = frames
                replayed["query"] = query
                return {"fire_laser": False, "result": {"status": "not_found", "reason": "no"}}

            run_object_search(
                2,
                "keys",
                update,
                capture=replay_capture,
                locate=replay_locate,
                replay_dir=saved,
            )
        self.assertEqual(replayed["query"], "keys")
        self.assertEqual(replayed["frames"][0]["image"], jpeg)
        self.assertTrue(any(item.startswith("Loading saved photo") for item in details))

    def test_camera_error_fails_during_scan(self):
        from camera import CameraError

        search = {"id": 1}

        def update(_search_id, **changes):
            search.update(changes)
            return True

        def capture(on_frame=None):
            raise CameraError("Could not reach the camera.")

        run_object_search(1, "keys", update, capture=capture, locate=lambda *_: None)
        self.assertEqual(search["stage"], "failed")
        self.assertEqual(search["failed_at"], "scanning")
        self.assertIn("camera", search["error"].lower())

    def test_not_found_does_not_claim_target(self):
        jpeg = tiny_jpeg()
        search = {"id": 1}

        def update(_search_id, **changes):
            search.update(changes)
            return True

        run_object_search(
            1,
            "keys",
            update,
            capture=lambda on_frame=None: [{"image": jpeg, "pan": 0, "tilt": 0, "timestamp": "t"}],
            locate=lambda *_: {
                "fire_laser": False,
                "aim": None,
                "result": {"status": "not_found", "reason": "Nothing matched.", "bbox_px": None},
                "frame_jpeg": jpeg,
            },
        )
        self.assertEqual(search["stage"], "failed")
        self.assertEqual(search["failed_at"], "searching")
        self.assertEqual(search["detection"]["status"], "not_found")

    def test_announcement_says_where_the_object_is(self):
        self.assertEqual(
            announcement("pink eraser", "On top of the blue tape roll."),
            "The pink eraser is on top of the blue tape roll.",
        )
        self.assertEqual(
            announcement("the blue water bottle", "next to the white energy drink can"),
            "The blue water bottle is next to the white energy drink can.",
        )
        self.assertEqual(announcement("mug", None), "Found the mug.")
        self.assertEqual(announcement("mug", "  "), "Found the mug.")
        self.assertIsNone(announcement(None, "next to the lamp"))
        self.assertEqual(
            announcement("blue water bottle", "next to the laptop", best_guess=True),
            "My best guess: the blue water bottle is next to the laptop.",
        )

    def test_ambiguous_keeps_best_guess_aim_without_laser(self):
        from vision.aim import laser_payload

        aim = {"azimuth_deg": -37.2, "elevation_deg": 1.9}
        payload = laser_payload(
            {"fire_laser": False, "aim": aim, "result": {"status": "ambiguous"}}
        )
        self.assertFalse(payload["fire_laser"])
        self.assertEqual(payload["aim"], aim)
        payload = laser_payload({"fire_laser": False, "aim": aim, "result": {"status": "not_found"}})
        self.assertIsNone(payload["aim"])

    def test_annotate_detection_draws_box(self):
        jpeg = tiny_jpeg()
        marked = annotate_detection(jpeg, [2, 2, 20, 18], "mug")
        image = cv2.imdecode(np.frombuffer(marked, dtype=np.uint8), cv2.IMREAD_COLOR)
        original = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertFalse(np.array_equal(image, original))

    def test_search_frames_adds_center_and_keeps_box(self):
        from vision.aim import search_frames

        jpeg = tiny_jpeg()
        raw = {
            "fire_laser": True,
            "aim": {"azimuth_deg": 1.0, "elevation_deg": 2.0},
            "result": {
                "status": "found",
                "object_id": "obj_001",
                "label": "mug",
                "bbox_px": [1, 2, 3, 4],
                "azimuth_deg": 1.0,
                "elevation_deg": 2.0,
            },
            "items": [{"id": "obj_001", "image_width": 40, "image_height": 30}],
        }
        with patch("vision.aim.answer", return_value=raw), patch(
            "vision.aim.require_api_key", return_value="x"
        ):
            located = search_frames([{"image": jpeg, "pan": 0, "tilt": 0}], "mug")
        self.assertEqual(located["result"]["bbox_px"], [1, 2, 3, 4])
        self.assertEqual(located["result"]["center_px"], [2, 3])
        self.assertEqual(located["result"]["image_width"], 40)
        self.assertTrue(located["frame_jpeg"].startswith(b"\xff\xd8"))


def found_located(jpeg, azimuth=-40.0, elevation=-10.0, status="found"):
    return {
        "fire_laser": status == "found",
        "aim": {"azimuth_deg": azimuth, "elevation_deg": elevation},
        "result": {
            "status": status,
            "label": "mug",
            "bbox_px": [8, 6, 28, 24],
            "center_px": [18, 15],
            "image_width": 40,
            "image_height": 30,
            "reason": "reason",
        },
        "frame_jpeg": jpeg,
    }


class TurretSearchTests(unittest.TestCase):
    def setUp(self):
        self.jpeg = tiny_jpeg()
        self.search = {"id": 1}
        self.turret = FakeTurret()
        self.timer = patch("object_search.schedule_laser_off")
        self.schedule = self.timer.start()
        self.addCleanup(self.timer.stop)
        scans = tempfile.TemporaryDirectory()
        self.addCleanup(scans.cleanup)
        self.scans = scans.name

    def update(self, _search_id, **changes):
        self.search.update(changes)
        return True

    def capture(self, on_frame=None, turret=None, should_continue=None):
        self.assertIs(turret, self.turret)
        self.assertTrue(should_continue())
        turret.laser_off()
        turret.home()
        pose = turret.goto(0, 0)
        return [{"image": self.jpeg, "pan": pose[0], "tilt": 0.0, "timestamp": "t"}]

    def run_search(self, located):
        run_object_search(
            1,
            "mug",
            self.update,
            capture=self.capture,
            locate=lambda *_args, **_kw: located,
            save_dir=self.scans,
            turret=self.turret,
        )

    def test_found_and_reachable_aims_then_fires(self):
        with patch.dict("os.environ", {"LASER_ON_SECONDS": "15"}):
            self.run_search(found_located(self.jpeg))
        self.assertEqual(self.search["stage"], "on_target")
        self.assertEqual(self.turret.calls[0], ("laser_off",))
        self.assertEqual(self.turret.calls[-2:], [("goto", -40.0, -10.0), ("laser_on",)])
        self.assertTrue(self.turret.laser)
        self.assertEqual(self.schedule.call_args.args[1], 15.0)

    def test_laser_seconds_zero_aims_without_firing(self):
        with patch.dict("os.environ", {"LASER_ON_SECONDS": "0"}):
            self.run_search(found_located(self.jpeg))
        self.assertEqual(self.search["stage"], "on_target")
        self.assertEqual(self.turret.calls[-1], ("goto", -40.0, -10.0))
        self.assertFalse(self.turret.laser)

    def test_not_found_never_aims(self):
        self.run_search(found_located(self.jpeg, status="not_found"))
        self.assertEqual(self.search["stage"], "failed")
        aims = [call for call in self.turret.calls if call == ("goto", -40.0, -10.0)]
        self.assertEqual(aims, [])
        self.assertNotIn(("laser_on",), self.turret.calls)

    def test_ambiguous_pans_to_best_guess_and_lights_laser(self):
        with patch.dict("os.environ", {"LASER_ON_SECONDS": "10"}):
            self.run_search(found_located(self.jpeg, status="ambiguous"))
        self.assertEqual(self.search["stage"], "on_target")
        self.assertTrue(self.search["best_guess"])
        self.assertEqual(self.search["detection"]["announcement"], "My best guess is the mug.")
        self.assertEqual(self.turret.calls[-2:], [("goto", -40.0, -10.0), ("laser_on",)])
        self.assertEqual(self.schedule.call_args.args[1], 10.0)

    def test_ambiguous_without_aim_still_fails(self):
        located = found_located(self.jpeg, status="ambiguous")
        located["aim"] = None
        self.run_search(located)
        self.assertEqual(self.search["stage"], "failed")
        self.assertNotIn(("goto", -40.0, -10.0), self.turret.calls)

    def test_out_of_reach_fails_without_laser(self):
        self.run_search(found_located(self.jpeg, azimuth=25.0))
        self.assertEqual(self.search["stage"], "failed")
        self.assertEqual(self.search["failed_at"], "pointing")
        self.assertNotIn(("laser_on",), self.turret.calls)

    def test_replaced_search_never_fires(self):
        live = {"current": True}

        def update(_search_id, **changes):
            if changes.get("stage") == "pointing":
                live["current"] = False
            if not live["current"]:
                return False
            self.search.update(changes)
            return True

        run_object_search(
            1,
            "mug",
            update,
            capture=self.capture,
            locate=lambda *_args, **_kw: found_located(self.jpeg),
            save_dir=self.scans,
            turret=self.turret,
        )
        self.assertNotIn(("laser_on",), self.turret.calls)

    def test_turret_error_during_scan_fails_scanning(self):
        def capture(on_frame=None, turret=None, should_continue=None):
            raise TurretError("Arduino not found on COM4.")

        run_object_search(1, "mug", self.update, capture=capture, locate=lambda *_: None, turret=self.turret)
        self.assertEqual(self.search["stage"], "failed")
        self.assertEqual(self.search["failed_at"], "scanning")
        self.assertIn("COM4", self.search["error"])

    def test_turret_error_while_aiming_turns_laser_off(self):
        original_goto = self.turret.goto

        def goto(pan, tilt):
            if (pan, tilt) == (-40.0, -10.0):
                raise TurretError("The turret did not answer.")
            return original_goto(pan, tilt)

        self.turret.goto = goto
        self.run_search(found_located(self.jpeg))
        self.assertEqual(self.search["stage"], "failed")
        self.assertEqual(self.search["failed_at"], "pointing")
        self.assertEqual(self.turret.calls[-1], ("laser_off",))


class SearchApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "accounts.sqlite3"
        self.config = patch.dict(app.config, TESTING=True, AUTH_DATABASE=self.database)
        self.config.start()
        self.client = app.test_client()
        self.client.post("/api/auth/register", json={
            "email": "person@example.com", "password": "test-password",
        })

    def tearDown(self):
        self.config.stop()
        self.directory.cleanup()

    def test_search_starts_scan_and_serves_preview(self):
        jpeg = tiny_jpeg()

        def fake_run(search_id, query, update):
            update(
                search_id,
                stage="on_target",
                photos_taken=7,
                photos_checked=7,
                total_photos=7,
                detection={
                    "status": "found",
                    "label": query,
                    "bbox_px": [1, 2, 3, 4],
                    "center_px": [2, 3],
                    "azimuth_deg": 1.5,
                    "elevation_deg": -0.5,
                    "image_width": 40,
                    "image_height": 30,
                },
                preview_jpeg=jpeg,
            )

        with patch("app.threading.Thread") as thread:
            response = self.client.post("/api/search", json={"query": "blue bottle"})
        self.assertIsNone(thread.call_args.kwargs["kwargs"]["replay_dir"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["stage"], "scanning")
        self.assertEqual(response.json["total_photos"], 7)
        self.assertFalse(response.json["has_preview"])

        import app as app_module
        fake_run(response.json["id"], "blue bottle", app_module._update)
        status = self.client.get(f"/api/search/{response.json['id']}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json["detection"]["bbox_px"], [1, 2, 3, 4])
        preview = self.client.get(f"/api/search/{response.json['id']}/preview")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, "image/jpeg")
        self.assertTrue(preview.data.startswith(b"\xff\xd8"))

        other = app.test_client()
        other.post("/api/auth/register", json={
            "email": "other@example.com", "password": "test-password",
        })
        self.assertEqual(other.get(f"/api/search/{response.json['id']}").status_code, 404)
        self.assertEqual(other.get(f"/api/search/{response.json['id']}/preview").status_code, 404)

    def test_scan_dir_replays_a_saved_folder(self):
        folder = Path(self.directory.name) / "scan"
        folder.mkdir()
        (folder / "manifest.json").write_text('{"frames": []}', encoding="utf-8")
        with patch("app.resolve_saved_scan", return_value=folder), patch("app.threading.Thread") as thread:
            response = self.client.post("/api/search", json={"query": "mug", "scan_dir": "scan"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["detail"], "Loading a saved scan…")
        self.assertEqual(thread.call_args.kwargs["kwargs"]["replay_dir"], folder)

        with patch("app.threading.Thread") as thread:
            rejected = self.client.post("/api/search", json={"query": "mug", "scan_dir": "../secret"})
        self.assertEqual(rejected.status_code, 400)
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
