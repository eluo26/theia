"""Query-scoped indexing and the single end-of-scan ranking pass."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from vision.config import load_settings
from vision.grok_client import index_prompt
from vision.index import filter_indexed_for_query, merge_detections, scan_fingerprint
from vision.matching import same_kind
from vision.query import _collapse_same_kind, _direct_catalog_decision, locate
from vision.schemas import (
    Candidate,
    Catalog,
    CatalogObject,
    Detection,
    IndexedObject,
    QueryDecision,
)


def _indexed(label, drug_name=None):
    return IndexedObject(
        label=label,
        description=label,
        box=[0.0, 0.0, 100.0, 100.0],
        count=1,
        drug_name=drug_name,
        expiry_text=None,
    )


def _object(
    object_id, label, confidence, azimuth, elevation=0.0, views=1, clear_views=1, bbox=(0, 0, 10, 10)
):
    return CatalogObject(
        object_id=object_id,
        label=label,
        description=label,
        count=1,
        views=views,
        clear_views=clear_views,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        confidence=confidence,
        frame_file="frame.jpg",
        bbox_px=list(bbox),
        pan_deg=0.0,
        tilt_deg=0.0,
        image_width=100,
        image_height=100,
    )


def _catalog(*objects):
    return Catalog(objects=list(objects), created_at="t")


_SETTINGS = SimpleNamespace(label_sim=85, merge_deg=8)


class IndexPromptTests(unittest.TestCase):
    def test_unscoped_prompt_lists_the_scene(self):
        prompt = index_prompt(12)
        self.assertEqual(prompt, index_prompt(12, query=None))
        self.assertEqual(prompt, index_prompt(12, query="  "))
        self.assertIn(
            "List the distinct objects a person might ask you to find. At most 12.",
            prompt,
        )
        self.assertNotIn("The user asked", prompt)

    def test_scoped_prompt_names_the_question(self):
        prompt = index_prompt(12, query="blue water bottle")
        self.assertIn('The user is looking for: "blue water bottle".', prompt)
        self.assertIn("could be what they asked for", prompt)
        self.assertNotEqual(prompt, index_prompt(12))


class FilterTests(unittest.TestCase):
    def test_keeps_plausible_labels_and_drops_unrelated_ones(self):
        objects = [
            _indexed("blue water bottle"),
            _indexed("black water bottle"),
            _indexed("plastic bottle"),
            _indexed("laptop"),
        ]
        kept = filter_indexed_for_query(objects, "blue water bottle", 85)
        self.assertEqual(
            [obj.label for obj in kept],
            ["blue water bottle", "black water bottle", "plastic bottle"],
        )

    def test_drug_name_can_keep_a_package(self):
        objects = [_indexed("pill bottle", drug_name="Jardiance"), _indexed("laptop")]
        kept = filter_indexed_for_query(objects, "Jardiance", 85)
        self.assertEqual([obj.label for obj in kept], ["pill bottle"])

    def test_category_question_keeps_the_model_list(self):
        objects = [_indexed("blue water bottle"), _indexed("laptop")]
        kept = filter_indexed_for_query(objects, "something to drink", 85)
        self.assertEqual([obj.label for obj in kept], ["blue water bottle", "laptop"])


class RankingTests(unittest.TestCase):
    def test_one_close_candidate_is_found(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(_object("obj_001", "blue water bottle", 0.8, 10)),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "found")
        self.assertEqual(decision.object_id, "obj_001")

    def test_color_conflict_is_not_a_second_match(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(
                _object("obj_001", "blue water bottle", 0.7, 0),
                _object("obj_002", "black water bottle", 0.95, 40),
            ),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "found")
        self.assertEqual(decision.object_id, "obj_001")

    def test_far_match_with_more_evidence_is_found(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(
                _object("obj_001", "blue water bottle", 0.9, 0),
                _object("obj_002", "blue water bottle", 0.6, 40),
            ),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "found")
        self.assertEqual(decision.object_id, "obj_001")

    def test_more_photos_beat_a_single_view(self):
        seen_often = _object("obj_001", "blue water bottle", 0.45, -80, views=4, clear_views=3)
        seen_once = _object("obj_002", "blue water bottle", 0.45, 30)
        decision = _direct_catalog_decision(
            "blue water bottle", _catalog(seen_once, seen_often), _SETTINGS
        )
        self.assertEqual(decision.status, "found")
        self.assertEqual(decision.object_id, "obj_001")

    def test_whole_box_beats_one_cut_off_by_the_edge(self):
        cut_off = _object("obj_001", "energy drink can", 0.45, -120, views=3, clear_views=0)
        whole = _object("obj_002", "energy drink can", 0.45, -40, views=2, clear_views=2)
        decision = _direct_catalog_decision(
            "energy drink can", _catalog(cut_off, whole), _SETTINGS
        )
        self.assertEqual(decision.object_id, "obj_002")

    def test_tied_far_matches_stay_ambiguous_best_first(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(
                _object("obj_001", "blue water bottle", 0.45, 40, bbox=[0, 0, 10, 10]),
                _object("obj_002", "blue water bottle", 0.45, 0, bbox=[45, 45, 55, 55]),
            ),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "ambiguous")
        self.assertIsNone(decision.object_id)
        # The centered view comes first, so the best guess aims at it.
        self.assertEqual(
            [item.object_id for item in decision.candidates],
            ["obj_002", "obj_001"],
        )

    def test_two_near_matches_pick_higher_confidence(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(
                _object("obj_001", "blue water bottle", 0.4, 0),
                _object("obj_002", "blue water bottle", 0.9, 4),
            ),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "found")
        self.assertEqual(decision.object_id, "obj_002")

    def test_text_ranking_keeps_distant_matches_ambiguous(self):
        objects = _catalog(
            _object("obj_001", "blue water bottle", 0.5, 0),
            _object("obj_002", "blue water bottle", 0.5, 40),
        )
        decision = QueryDecision(
            status="ambiguous",
            object_id=None,
            confidence=0.5,
            reason="Two bottles.",
            candidates=[
                Candidate(object_id="obj_001", label="blue water bottle", confidence=0.5, reason="a"),
                Candidate(object_id="obj_002", label="blue water bottle", confidence=0.8, reason="b"),
            ],
        )
        collapsed = _collapse_same_kind(decision, decision.candidates, objects, _SETTINGS)
        self.assertEqual(collapsed.status, "ambiguous")
        self.assertIsNone(collapsed.object_id)

    def test_text_ranking_collapses_nearby_matches(self):
        objects = _catalog(
            _object("obj_001", "blue water bottle", 0.4, 0),
            _object("obj_002", "blue water bottle", 0.9, 3),
        )
        decision = QueryDecision(
            status="ambiguous",
            object_id=None,
            confidence=0.5,
            reason="Two views.",
            candidates=[
                Candidate(object_id="obj_001", label="blue water bottle", confidence=0.2, reason="a"),
                Candidate(object_id="obj_002", label="blue water bottle", confidence=0.8, reason="b"),
            ],
        )
        collapsed = _collapse_same_kind(decision, decision.candidates, objects, _SETTINGS)
        self.assertEqual(collapsed.status, "found")
        self.assertEqual(collapsed.object_id, "obj_002")

    def test_text_ranking_collapses_one_kind_split_across_photos(self):
        objects = _catalog(
            _object("obj_001", "black computer mouse", 0.45, -175, views=1, clear_views=0),
            _object("obj_002", "black computer mouse", 0.45, -80, views=2, clear_views=2),
            _object("obj_003", "black mouse", 0.45, -30, views=1, clear_views=1),
        )
        decision = QueryDecision(
            status="ambiguous",
            object_id=None,
            confidence=0.5,
            reason="Several mice.",
            candidates=[
                Candidate(object_id=obj.object_id, label=obj.label, confidence=0.5, reason="a")
                for obj in objects.objects
            ],
        )
        collapsed = _collapse_same_kind(decision, decision.candidates, objects, _SETTINGS)
        self.assertEqual(collapsed.status, "found")
        self.assertEqual(collapsed.object_id, "obj_002")

    def test_text_ranking_keeps_different_kinds_ambiguous(self):
        objects = _catalog(
            _object("obj_001", "red apple", 0.45, 0),
            _object("obj_002", "orange", 0.45, 40, views=3, clear_views=3),
        )
        decision = QueryDecision(
            status="ambiguous",
            object_id=None,
            confidence=0.5,
            reason="Both are fruit.",
            candidates=[
                Candidate(object_id="obj_001", label="red apple", confidence=0.6, reason="a"),
                Candidate(object_id="obj_002", label="orange", confidence=0.5, reason="b"),
            ],
        )
        collapsed = _collapse_same_kind(decision, decision.candidates, objects, _SETTINGS)
        self.assertEqual(collapsed, decision)


class SameKindTests(unittest.TestCase):
    def test_rewordings_of_one_object_are_one_kind(self):
        self.assertTrue(same_kind("blue water bottle", "blue bottle", 85))
        self.assertTrue(same_kind("L-Carnitine energy drink can", "silver energy drink can", 85))
        self.assertTrue(same_kind("computer mouse", "black computer mouse", 85))
        self.assertTrue(same_kind("water bottles", "blue water bottle", 85))

    def test_different_objects_or_colors_are_not(self):
        self.assertFalse(same_kind("black energy drink can", "white energy drink can", 85))
        self.assertFalse(same_kind("blue water bottle", "blue notebook", 85))


def _detection(label, frame, bbox, confidence, azimuth):
    return Detection(
        label=label,
        description=label,
        bbox_px=list(bbox),
        count=1,
        confidence=confidence,
        box_source="grok",
        frame_file=frame,
        pan_deg=0.0,
        tilt_deg=0.0,
        azimuth_deg=azimuth,
        elevation_deg=0.0,
        image_width=320,
        image_height=240,
    )


def _translated_scan(directory: Path):
    """Two overlapping photos of one room. The camera shifts right by 48 pixels."""
    rng = np.random.default_rng(4)
    canvas = rng.integers(0, 255, (240, 420, 3), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    cv2.rectangle(canvas, (90, 70), (160, 170), (40, 180, 40), -1)
    cv2.rectangle(canvas, (250, 40), (310, 120), (180, 40, 40), -1)
    previous = canvas[:, 0:320]
    current = canvas[:, 48:368]
    cv2.imwrite(str(directory / "frame_000.jpg"), previous)
    cv2.imwrite(str(directory / "frame_001.jpg"), current)
    speaker_previous = [90, 70, 160, 170]
    speaker_current = [42, 70, 112, 170]
    lamp_previous = [250, 40, 310, 120]
    lamp_current = [202, 40, 262, 120]
    return speaker_previous, speaker_current, lamp_previous, lamp_current


class OverlapMergeTests(unittest.TestCase):
    def test_repeated_views_collapse_to_the_confident_angle(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            speaker_previous, speaker_current, lamp_previous, lamp_current = _translated_scan(folder)
            detections = [
                _detection("speaker", "frame_000.jpg", speaker_previous, 0.4, -20),
                _detection("black speaker", "frame_001.jpg", speaker_current, 0.92, 15),
                _detection("lamp", "frame_000.jpg", lamp_previous, 0.5, 25),
                _detection("lamp", "frame_001.jpg", lamp_current, 0.55, 40),
            ]
            objects = merge_detections(detections, _SETTINGS, created_at="t", scan_dir=folder)
        labels = sorted(obj.label for obj in objects)
        self.assertEqual(labels, ["black speaker", "lamp"])
        speaker = next(obj for obj in objects if "speaker" in obj.label)
        self.assertEqual(speaker.azimuth_deg, 15)
        self.assertEqual(speaker.confidence, 0.92)

    def test_two_shifted_speakers_stay_two_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rng = np.random.default_rng(4)
            canvas = rng.integers(0, 255, (240, 460, 3), dtype=np.uint8)
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            cv2.rectangle(canvas, (70, 60), (130, 150), (40, 180, 40), -1)
            cv2.rectangle(canvas, (220, 50), (290, 160), (40, 180, 40), -1)
            shift = 48
            cv2.imwrite(str(folder / "frame_000.jpg"), canvas[:, 0:320])
            cv2.imwrite(str(folder / "frame_001.jpg"), canvas[:, shift : shift + 320])
            detections = [
                _detection("speaker", "frame_000.jpg", [70, 60, 130, 150], 0.4, -25),
                _detection("speaker", "frame_000.jpg", [220, 50, 290, 160], 0.5, 20),
                _detection("speaker", "frame_001.jpg", [22, 60, 82, 150], 0.91, -8),
                _detection("speaker", "frame_001.jpg", [172, 50, 242, 160], 0.7, 33),
            ]
            objects = merge_detections(detections, _SETTINGS, created_at="t", scan_dir=folder)
        self.assertEqual(len(objects), 2)
        self.assertEqual(sorted(obj.azimuth_deg for obj in objects), [-8, 33])

    def test_a_single_object_still_links_when_the_wall_has_no_features(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            blank = np.full((80, 80, 3), 20, np.uint8)
            cv2.imwrite(str(folder / "frame_000.jpg"), blank)
            cv2.imwrite(str(folder / "frame_001.jpg"), blank)
            detections = [
                _detection("speaker", "frame_000.jpg", [10, 10, 40, 50], 0.3, -30),
                _detection("speaker", "frame_001.jpg", [30, 12, 60, 52], 0.8, 12),
            ]
            objects = merge_detections(detections, _SETTINGS, created_at="t", scan_dir=folder)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].azimuth_deg, 12)

    def test_one_object_per_photo_chains_when_pan_tags_are_wrong(self):
        # A slipping turret tags each photo with the wrong pan, so the same bottle
        # gets far-apart angles and the reference points do not place its box.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _translated_scan(folder)
            cv2.imwrite(str(folder / "frame_002.jpg"), cv2.imread(str(folder / "frame_001.jpg")))
            detections = [
                _detection("blue water bottle", "frame_000.jpg", [60, 20, 110, 200], 0.45, 33),
                _detection("blue bottle", "frame_001.jpg", [200, 20, 250, 200], 0.45, -7),
                _detection("blue water bottle", "frame_002.jpg", [0, 20, 40, 200], 0.45, -137),
            ]
            objects = merge_detections(detections, _SETTINGS, created_at="t", scan_dir=folder)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].views, 3)
        # frame_002 is cut off by the left edge, so it is not a clear view.
        self.assertEqual(objects[0].clear_views, 2)

    def test_two_instances_in_one_photo_stay_separate_without_features(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            blank = np.full((80, 80, 3), 20, np.uint8)
            cv2.imwrite(str(folder / "frame_000.jpg"), blank)
            cv2.imwrite(str(folder / "frame_001.jpg"), blank)
            detections = [
                _detection("speaker", "frame_000.jpg", [5, 10, 25, 40], 0.4, -40),
                _detection("speaker", "frame_000.jpg", [50, 10, 75, 40], 0.7, 40),
                _detection("speaker", "frame_001.jpg", [8, 12, 28, 42], 0.5, -22),
                _detection("speaker", "frame_001.jpg", [48, 12, 72, 42], 0.6, 22),
            ]
            objects = merge_detections(detections, _SETTINGS, created_at="t", scan_dir=folder)
        self.assertEqual(len(objects), 4)


class LocationTests(unittest.TestCase):
    def test_prompt_asks_where_the_object_sits(self):
        self.assertIn("- location:", index_prompt(12, query="eraser"))

    def test_merge_keeps_a_location_from_any_view(self):
        first = _detection("eraser", "frame_000.jpg", [0, 0, 10, 10], 0.9, 10)
        second = _detection("eraser", "frame_001.jpg", [0, 0, 10, 10], 0.4, 12)
        second.location = "on top of the blue tape roll"
        objects = merge_detections([first, second], _SETTINGS, created_at="t")
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].location, "on top of the blue tape roll")

    def test_found_result_carries_the_location(self):
        eraser = _object("obj_001", "pink eraser", 0.8, -30)
        eraser.location = "on top of the blue tape roll"
        with patch("vision.query.require_api_key", return_value="x"):
            result = locate("pink eraser", _catalog(eraser), settings=load_settings())
        self.assertEqual(result.status, "found")
        self.assertEqual(result.location, "on top of the blue tape roll")

    def test_old_catalog_without_location_still_loads(self):
        payload = _object("obj_001", "mug", 0.8, 0).model_dump()
        payload.pop("location")
        self.assertIsNone(CatalogObject.model_validate(payload).location)


class FingerprintTests(unittest.TestCase):
    def test_query_changes_the_catalog_fingerprint(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "a.jpg").write_bytes(b"jpeg")
            bare = scan_fingerprint(path, settings, sweep=None, angles_csv=None)
            blue = scan_fingerprint(
                path, settings, sweep=None, angles_csv=None, query="blue water bottle"
            )
            keys = scan_fingerprint(path, settings, sweep=None, angles_csv=None, query="keys")
        self.assertNotEqual(bare, blue)
        self.assertNotEqual(blue, keys)


if __name__ == "__main__":
    unittest.main()
