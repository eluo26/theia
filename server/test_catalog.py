"""Query-scoped indexing and the single end-of-scan ranking pass."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vision.config import load_settings
from vision.grok_client import index_prompt
from vision.index import filter_indexed_for_query, scan_fingerprint
from vision.query import _collapse_same_kind, _direct_catalog_decision
from vision.schemas import Candidate, Catalog, CatalogObject, IndexedObject, QueryDecision


def _indexed(label, drug_name=None):
    return IndexedObject(
        label=label,
        description=label,
        box=[0.0, 0.0, 100.0, 100.0],
        count=1,
        drug_name=drug_name,
        expiry_text=None,
    )


def _object(object_id, label, confidence, azimuth, elevation=0.0):
    return CatalogObject(
        object_id=object_id,
        label=label,
        description=label,
        count=1,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        confidence=confidence,
        frame_file="frame.jpg",
        bbox_px=[0, 0, 10, 10],
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

    def test_two_far_matches_stay_ambiguous(self):
        decision = _direct_catalog_decision(
            "blue water bottle",
            _catalog(
                _object("obj_001", "blue water bottle", 0.9, 0),
                _object("obj_002", "blue water bottle", 0.6, 40),
            ),
            _SETTINGS,
        )
        self.assertEqual(decision.status, "ambiguous")
        self.assertIsNone(decision.object_id)
        self.assertEqual(
            [item.object_id for item in decision.candidates],
            ["obj_001", "obj_002"],
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
