import unittest

import cv2
import numpy as np

from laser import DotNotFound, correction, find_laser_dot


class LaserTests(unittest.TestCase):
    def test_direction_distance_and_stop_boundary(self):
        result = correction((100, 100), (60, 90))
        self.assertEqual((result.dx_px, result.dy_px), (-40, -10))
        self.assertAlmostEqual(result.distance_px, 41.231056, places=5)
        self.assertIn('left', result.instruction)
        self.assertIn('up', result.instruction)
        self.assertTrue(correction((0, 0), (3, 4)).on_target)
        self.assertFalse(correction((0, 0), (3, 4.01)).on_target)
        with self.assertRaises(ValueError):
            correction((float('nan'), 0), (0, 0))

    def test_new_dot_ignores_existing_red_object(self):
        off = np.full((160, 200, 3), 100, np.uint8)
        cv2.circle(off, (30, 40), 8, (0, 0, 255), -1)
        on = off.copy()
        cv2.circle(on, (120, 90), 4, (20, 20, 255), -1)
        point, _ = find_laser_dot(off, on, align=False)
        np.testing.assert_allclose(point, (120, 90), atol=0.2)
        with self.assertRaises(DotNotFound):
            find_laser_dot(off, off, align=False)
        with self.assertRaises(DotNotFound):
            find_laser_dot(on, off, align=False)

    def test_ambiguous_spots_rejected(self):
        off = np.full((160, 200, 3), 100, np.uint8)
        on = off.copy()
        for point in ((40, 50), (120, 90)):
            cv2.circle(on, point, 4, (0, 0, 255), -1)
        with self.assertRaises(DotNotFound):
            find_laser_dot(off, on, align=False)

    def test_camera_shift_preserves_on_image_coordinates(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(40, 180, (400, 500), dtype=np.uint8)
        off = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        on = cv2.warpAffine(off, np.float32([[1, 0, 12], [0, 1, -8]]), (500, 400))
        cv2.circle(on, (250, 230), 5, (20, 20, 255), -1)
        point, diagnostics = find_laser_dot(off, on)
        np.testing.assert_allclose(point, (250, 230), atol=0.5)
        self.assertGreater(diagnostics['alignment']['inliers'], 10)


if __name__ == '__main__':
    unittest.main()
