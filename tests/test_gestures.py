"""
Unit tests for gestures.py. No webcam, OpenCV or MediaPipe needed.

Run from the project folder:
    python -m unittest discover tests -v

The hands below are synthetic: 21 hand-built landmark positions that mimic
MediaPipe's layout. They test the rules and the smoothing logic, not
real-world recognition accuracy.
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gestures as g  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sample_hand_landmarks.json")

# Upright right hand, palm facing the camera, image coordinates (y points down).
MCPS = {"index": (-0.35, -0.95), "middle": (0.0, -1.0), "ring": (0.3, -0.95), "pinky": (0.55, -0.85)}
THUMB_CMC = (-0.3, -0.25)
THUMB_MCP = (-0.55, -0.45)


def _finger(mcp, state):
    x, y = mcp
    extended = [(x, y), (x, y - 0.45), (x, y - 0.75), (x, y - 1.0)]
    curled = [(x, y), (x, y - 0.4), (x, y - 0.15), (x, y + 0.05)]
    if state == "up":
        joints = extended
    elif state == "down":
        joints = curled
    else:  # "half": halfway between straight and curled
        joints = [((ax + bx) / 2, (ay + by) / 2) for (ax, ay), (bx, by) in zip(extended, curled)]
    return [(px, py, 0.0) for px, py in joints]


def _thumb(state):
    mx, my = THUMB_MCP
    if state == "out":
        dx, dy = -1 / math.hypot(1, 0.3), -0.3 / math.hypot(1, 0.3)
        ip, tip = (mx + dx * 0.35, my + dy * 0.35), (mx + dx * 0.65, my + dy * 0.65)
    else:  # "in": tucked across the palm
        ip, tip = (mx + 0.35, my - 0.15), (mx + 0.65, my - 0.15)
    return [(*THUMB_CMC, 0.0), (*THUMB_MCP, 0.0), (*ip, 0.0), (*tip, 0.0)]


def make_hand(thumb="in", index="down", middle="down", ring="down", pinky="down"):
    points = [(0.0, 0.0, 0.0)] + _thumb(thumb)
    for name, state in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        points += _finger(MCPS[name], state)
    return points


def make_ok_hand():
    points = make_hand(thumb="out", middle="up", ring="up", pinky="up")
    # Bend the index finger so its tip meets the thumb tip.
    points[6], points[7], points[8] = (-0.35, -1.35, 0.0), (-0.55, -1.4, 0.0), (-0.65, -1.2, 0.0)
    points[3], points[4] = (-0.62, -0.85, 0.0), (-0.62, -1.18, 0.0)
    return points


def rotate(points, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s, x * s + y * c, z) for x, y, z in points]


def transform(points, scale=1.0, dx=0.0, dy=0.0, mirror=False):
    return [((-x if mirror else x) * scale + dx, y * scale + dy, z * scale) for x, y, z in points]


EXAMPLES = {
    "OPEN_PALM": make_hand(thumb="out", index="up", middle="up", ring="up", pinky="up"),
    "FIST": make_hand(),
    # A thumbs-up fist is held sideways: rotate so the thumb points up/down.
    "THUMBS_UP": rotate(make_hand(thumb="out"), 90),
    "THUMBS_DOWN": rotate(make_hand(thumb="out"), -90),
    "POINT": make_hand(index="up"),
    "VICTORY": make_hand(index="up", middle="up"),
    "OK": make_ok_hand(),
    "CALL_ME": make_hand(thumb="out", pinky="up"),
    "I_LOVE_YOU": make_hand(thumb="out", index="up", pinky="up"),
    "L_SHAPE": make_hand(thumb="out", index="up"),
    "THREE": make_hand(index="up", middle="up", ring="up"),
    "FOUR": make_hand(index="up", middle="up", ring="up", pinky="up"),
    "PINKY": make_hand(pinky="up"),
    "ROCK": make_hand(index="up", pinky="up"),
}


class ClassifierTests(unittest.TestCase):
    def test_every_supported_gesture_has_an_example(self):
        self.assertEqual(set(EXAMPLES), set(g.GESTURE_WORDS))
        self.assertEqual(set(g.GESTURE_HINTS), set(g.GESTURE_WORDS))

    def test_each_example_is_recognised(self):
        for name, points in EXAMPLES.items():
            with self.subTest(gesture=name):
                gesture, confidence = g.classify(points)
                self.assertEqual(gesture, name)
                self.assertGreaterEqual(confidence, g.MIN_CONFIDENCE)

    def test_scale_and_position_do_not_matter(self):
        for name, points in EXAMPLES.items():
            with self.subTest(gesture=name):
                moved = transform(points, scale=180.0, dx=320.0, dy=400.0)
                self.assertEqual(g.classify(moved)[0], name)

    def test_left_and_right_hands_both_work(self):
        for name, points in EXAMPLES.items():
            with self.subTest(gesture=name):
                self.assertEqual(g.classify(transform(points, mirror=True))[0], name)

    def test_tilted_hand_still_recognised(self):
        for name in ("OPEN_PALM", "FIST", "POINT", "VICTORY", "CALL_ME", "I_LOVE_YOU", "OK",
                     "L_SHAPE", "THREE", "FOUR", "PINKY", "ROCK"):
            for angle in (-25, 25):
                with self.subTest(gesture=name, angle=angle):
                    self.assertEqual(g.classify(rotate(EXAMPLES[name], angle))[0], name)

    def test_sideways_thumb_is_not_thumbs_up_or_down(self):
        # Fist with thumb out, held upright: the thumb points sideways.
        gesture, _ = g.classify(make_hand(thumb="out"))
        self.assertEqual(gesture, g.UNKNOWN)

    def test_unsupported_shape_is_unknown(self):
        # Middle + ring up only is not in the vocabulary.
        self.assertEqual(g.classify(make_hand(middle="up", ring="up"))[0], g.UNKNOWN)

    def test_half_bent_fingers_are_uncertain(self):
        hand = make_hand(thumb="out", index="half", middle="half", ring="half", pinky="half")
        self.assertEqual(g.classify(hand)[0], g.UNKNOWN)

    def test_bad_input_is_unknown(self):
        self.assertEqual(g.classify([(0, 0, 0)] * 21), (g.UNKNOWN, 0.0))
        self.assertEqual(g.classify([(0, 0, 0)] * 5), (g.UNKNOWN, 0.0))

    def test_normalize_puts_wrist_at_origin_with_unit_palm(self):
        pts = g.normalize_landmarks(transform(EXAMPLES["OPEN_PALM"], scale=50, dx=10, dy=20))
        self.assertEqual(pts[g.WRIST], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(math.dist(pts[g.WRIST], pts[g.MIDDLE[0]]), 1.0)

    def test_normalize_rejects_wrong_landmark_count(self):
        with self.assertRaises(ValueError):
            g.normalize_landmarks([(0, 0, 0)] * 20)


class RealLandmarkTests(unittest.TestCase):
    """Landmarks MediaPipe produced on its own public sample photos (see the fixture's _source)."""

    def test_sample_photos(self):
        with open(FIXTURES) as f:
            hands = json.load(f)["hands"]
        for name, hand in hands.items():
            with self.subTest(photo=name):
                self.assertEqual(g.classify(hand["points"])[0], hand["expected"])
                mirrored = [(-x, y, z) for x, y, z in hand["points"]]
                self.assertEqual(g.classify(mirrored)[0], hand["expected"])


class SmootherTests(unittest.TestCase):
    def test_needs_a_majority_before_becoming_stable(self):
        s = g.GestureSmoother(window=10, min_votes=7)
        results = [s.update("FIST", 0.9) for _ in range(7)]
        self.assertEqual(results[:6], [None] * 6)
        self.assertEqual(results[6], "FIST")

    def test_flicker_never_becomes_stable(self):
        s = g.GestureSmoother(window=10, min_votes=7)
        for i in range(30):
            stable = s.update("FIST" if i % 2 else "OPEN_PALM", 0.9)
            self.assertIsNone(stable)

    def test_low_confidence_frames_do_not_count(self):
        s = g.GestureSmoother(window=10, min_votes=7, min_confidence=0.6)
        for _ in range(10):
            self.assertIsNone(s.update("FIST", 0.4))

    def test_losing_the_hand_clears_the_stable_gesture(self):
        s = g.GestureSmoother(window=10, min_votes=7)
        for _ in range(10):
            s.update("FIST", 0.9)
        self.assertEqual(s.stable, "FIST")
        for _ in range(4):
            s.update(None, 0.0)
        self.assertIsNone(s.stable)

    def test_tolerates_occasional_noise(self):
        s = g.GestureSmoother(window=10, min_votes=7)
        seq = ["POINT"] * 4 + [g.UNKNOWN] + ["POINT"] * 3 + ["VICTORY"] + ["POINT"]
        for gesture in seq:
            stable = s.update(gesture, 0.9)
        self.assertEqual(stable, "POINT")

    def test_reset(self):
        s = g.GestureSmoother(window=4, min_votes=3)
        for _ in range(4):
            s.update("FIST", 0.9)
        s.reset()
        self.assertIsNone(s.stable)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            g.GestureSmoother(window=5, min_votes=6)


class HoldToAddTests(unittest.TestCase):
    def test_fires_once_after_hold_time(self):
        h = g.HoldToAdd(hold_seconds=1.5)
        self.assertIsNone(h.update("FIST", 0.0))
        self.assertIsNone(h.update("FIST", 1.0))
        self.assertEqual(h.update("FIST", 1.6), "FIST")
        # Keeping the hand still must not add it again.
        for t in (1.7, 3.0, 10.0):
            self.assertIsNone(h.update("FIST", t))

    def test_same_gesture_again_after_hand_leaves(self):
        h = g.HoldToAdd(hold_seconds=1.0)
        h.update("FIST", 0.0)
        self.assertEqual(h.update("FIST", 1.0), "FIST")
        h.update(None, 1.5)
        h.update("FIST", 2.0)
        self.assertEqual(h.update("FIST", 3.0), "FIST")

    def test_changing_gesture_restarts_the_timer(self):
        h = g.HoldToAdd(hold_seconds=1.0)
        h.update("FIST", 0.0)
        h.update("POINT", 0.9)
        self.assertIsNone(h.update("POINT", 1.5))
        self.assertEqual(h.update("POINT", 2.0), "POINT")

    def test_no_hand_never_fires(self):
        h = g.HoldToAdd(hold_seconds=0.5)
        for t in range(5):
            self.assertIsNone(h.update(None, float(t)))

    def test_manual_add_prevents_auto_add_of_same_hold(self):
        h = g.HoldToAdd(hold_seconds=1.0)
        h.update("VICTORY", 0.0)
        h.mark_used()  # user pressed Add / Space
        self.assertIsNone(h.update("VICTORY", 2.0))
        self.assertEqual(h.progress(2.0), 0.0)

    def test_progress(self):
        h = g.HoldToAdd(hold_seconds=2.0)
        h.update("FIST", 0.0)
        self.assertAlmostEqual(h.progress(1.0), 0.5)
        h.update("FIST", 2.0)
        self.assertEqual(h.progress(2.5), 0.0)


if __name__ == "__main__":
    unittest.main()
