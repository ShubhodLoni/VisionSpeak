"""
Landmark-based hand-gesture classifier for VisionSpeak.

This is a small, transparent *prototype*: a handful of static one-hand
gestures are recognised from the 21 MediaPipe hand landmarks using simple
geometric rules. It is NOT a sign-language translator. Real sign languages
also use movement, facial expression, body position and grammar, none of
which are modelled here.

This module only uses the Python standard library so it can be unit-tested
without a webcam, OpenCV or MediaPipe.

Landmark indices (MediaPipe hand model):
    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring   (MCP, PIP, DIP, TIP)
    17-20 pinky  (MCP, PIP, DIP, TIP)
"""

import math
from collections import Counter, deque

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

UNKNOWN = "UNKNOWN"

# Supported gestures and the word/phrase each one adds to the sentence.
# These are VisionSpeak prototype gestures, not official ASL/ISL signs
# (except the "I love you" handshape, which matches the common ASL ILY sign).
# Edit the words here to customise the vocabulary.
GESTURE_WORDS = {
    "OPEN_PALM": "Hello",
    "FIST": "Stop",
    "THUMBS_UP": "Yes",
    "THUMBS_DOWN": "No",
    "POINT": "You",
    "VICTORY": "Peace",
    "OK": "OK",
    "CALL_ME": "Call me",
    "I_LOVE_YOU": "I love you",
    "L_SHAPE": "Help",
    "THREE": "Water",
    "FOUR": "Thank you",
    "PINKY": "I",
    "ROCK": "Awesome",
}

# Short descriptions shown in the UI and README.
GESTURE_HINTS = {
    "OPEN_PALM": "All five fingers spread open",
    "FIST": "All fingers curled, thumb tucked",
    "THUMBS_UP": "Fist with thumb pointing up",
    "THUMBS_DOWN": "Fist with thumb pointing down",
    "POINT": "Only the index finger extended",
    "VICTORY": "Index + middle finger extended (V)",
    "OK": "Thumb tip touches index tip, other fingers up",
    "CALL_ME": "Thumb + little finger extended",
    "I_LOVE_YOU": "Thumb + index + little finger extended",
    "L_SHAPE": "Thumb + index finger extended (L)",
    "THREE": "Index + middle + ring finger extended (W)",
    "FOUR": "Four fingers up, thumb tucked into palm",
    "PINKY": "Only the little finger extended",
    "ROCK": "Index + little finger, thumb tucked",
}

# ---------------------------------------------------------------- thresholds
# Finger: distance(wrist, TIP) / distance(wrist, PIP). A curled finger folds
# its tip back towards the wrist (ratio below ~1.0); a straight finger reaches
# well past its middle joint (ratio ~1.3+). In between counts as partially
# extended and lowers the confidence.
FINGER_CURLED_AT = 1.0
FINGER_EXTENDED_AT = 1.25

# Thumb: distance(thumb TIP, little-finger MCP) / distance(thumb MCP, little-finger MCP).
# A thumb tucked across the palm brings its tip closer to the little finger.
THUMB_TUCKED_AT = 1.0
THUMB_EXTENDED_AT = 1.25

# OK sign: thumb tip to index tip distance, divided by palm size.
PINCH_CLOSED_AT = 0.25
PINCH_OPEN_AT = 0.45

# Thumbs up/down: how vertical the thumb must point (0 = sideways, 1 = vertical).
THUMB_VERTICAL_AT = 0.5

# Minimum classifier confidence for a frame to count as a vote.
MIN_CONFIDENCE = 0.6


def _dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def _ramp(value, low, high):
    """Map value to 0..1, 0 at/below `low`, 1 at/above `high`."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def normalize_landmarks(points):
    """
    Translate landmarks so the wrist is at the origin and scale them so the
    palm (wrist -> middle-finger MCP) has length 1.

    `points` is a sequence of 21 (x, y, z) tuples in any consistent unit with
    y pointing down (image convention). Returns a list of 21 tuples.
    Raises ValueError for malformed input or a degenerate (zero-size) hand.
    """
    if len(points) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(points)}")
    pts = [tuple(float(c) for c in p[:3]) + (0.0,) * (3 - len(p[:3])) for p in points]
    origin = pts[WRIST]
    shifted = [tuple(c - o for c, o in zip(p, origin)) for p in pts]
    palm = _dist(shifted[WRIST], shifted[MIDDLE[0]])
    if palm < 1e-6:
        raise ValueError("degenerate hand: palm size is zero")
    return [tuple(c / palm for c in p) for p in shifted]


def finger_extension(points, finger):
    """0..1 score of how extended a non-thumb finger is (1 = straight)."""
    _mcp, pip, _dip, tip = (points[i] for i in finger)
    pip_reach = _dist(points[WRIST], pip)
    if pip_reach < 1e-6:
        return 0.0
    return _ramp(_dist(points[WRIST], tip) / pip_reach, FINGER_CURLED_AT, FINGER_EXTENDED_AT)


def thumb_extension(points):
    """0..1 score of how far the thumb sticks out from the palm."""
    pinky_mcp = points[PINKY[0]]
    base = _dist(points[THUMB[1]], pinky_mcp)
    if base < 1e-6:
        return 0.0
    return _ramp(_dist(points[THUMB[3]], pinky_mcp) / base, THUMB_TUCKED_AT, THUMB_EXTENDED_AT)


def thumb_vertical_direction(points):
    """
    How vertically the thumb points: +1 straight up, -1 straight down,
    near 0 when it points sideways. Uses image orientation (y down).
    """
    mcp, tip = points[THUMB[1]], points[THUMB[3]]
    dx, dy = tip[0] - mcp[0], tip[1] - mcp[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return -dy / length


def extract_features(points):
    """Normalise landmarks and compute the per-finger features used by the rules."""
    pts = normalize_landmarks(points)
    features = {name: finger_extension(pts, idx) for name, idx in FINGERS.items()}
    features["thumb"] = thumb_extension(pts)
    features["thumb_dir"] = thumb_vertical_direction(pts)
    features["pinch"] = _dist(pts[THUMB[3]], pts[INDEX[3]])
    return features


# Required state for each gesture: True = extended, False = curled.
# Fingers not listed are ignored by that gesture.
_SHAPES = {
    "OPEN_PALM": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
    "FIST": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
    "THUMBS_UP": {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False},
    "THUMBS_DOWN": {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False},
    "POINT": {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False},
    "VICTORY": {"index": True, "middle": True, "ring": False, "pinky": False},
    "OK": {"middle": True, "ring": True, "pinky": True},
    "CALL_ME": {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": True},
    "I_LOVE_YOU": {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True},
    "L_SHAPE": {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": False},
    "THREE": {"index": True, "middle": True, "ring": True, "pinky": False},
    "FOUR": {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": True},
    "PINKY": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": True},
    "ROCK": {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": True},
}


def _shape_score(features, shape):
    """Confidence that the hand matches `shape`: the weakest finger match wins."""
    scores = []
    for finger, extended in shape.items():
        value = features[finger]
        scores.append(value if extended else 1.0 - value)
    return min(scores)


def score_gestures(features):
    """Return {gesture: confidence 0..1} for every supported gesture."""
    scores = {name: _shape_score(features, shape) for name, shape in _SHAPES.items()}

    # Thumbs up / down additionally need the thumb to point vertically.
    up = _ramp(features["thumb_dir"], 0.0, THUMB_VERTICAL_AT)
    down = _ramp(-features["thumb_dir"], 0.0, THUMB_VERTICAL_AT)
    scores["THUMBS_UP"] = min(scores["THUMBS_UP"], up)
    scores["THUMBS_DOWN"] = min(scores["THUMBS_DOWN"], down)

    # OK needs the thumb and index tips touching. A touching pinch also rules
    # out OPEN_PALM, which would otherwise compete when the index is half bent.
    pinch_closed = 1.0 - _ramp(features["pinch"], PINCH_CLOSED_AT, PINCH_OPEN_AT)
    scores["OK"] = min(scores["OK"], pinch_closed)
    scores["OPEN_PALM"] = min(scores["OPEN_PALM"], 1.0 - pinch_closed)
    return scores


def classify(points, min_confidence=MIN_CONFIDENCE):
    """
    Classify one hand. Returns (gesture, confidence).

    gesture is UNKNOWN when no rule matches confidently enough, or when two
    gestures match almost equally well (ambiguous hand shape).
    """
    try:
        features = extract_features(points)
    except ValueError:
        return UNKNOWN, 0.0
    ranked = sorted(score_gestures(features).items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < min_confidence or best_score - runner_up < 0.15:
        return UNKNOWN, best_score
    return best, best_score


class GestureSmoother:
    """
    Temporal smoothing: a gesture only becomes "stable" once it wins a clear
    majority of the last `window` frames. Low-confidence frames and frames
    with no hand count as UNKNOWN votes, so flickering predictions never
    reach the sentence.
    """

    def __init__(self, window=10, min_votes=7, min_confidence=MIN_CONFIDENCE):
        if min_votes > window:
            raise ValueError("min_votes cannot exceed window")
        self.window = window
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self._votes = deque(maxlen=window)

    def update(self, gesture, confidence):
        """Add one frame's prediction; return the stable gesture or None."""
        if gesture is None or confidence < self.min_confidence:
            gesture = UNKNOWN
        self._votes.append(gesture)
        return self.stable

    @property
    def stable(self):
        if not self._votes:
            return None
        gesture, count = Counter(self._votes).most_common(1)[0]
        if gesture == UNKNOWN or count < self.min_votes:
            return None
        return gesture

    def reset(self):
        self._votes.clear()


class HoldToAdd:
    """
    Optional hands-free confirmation: a stable gesture held for `hold_seconds`
    is emitted once. It is not emitted again until the gesture changes or the
    hand leaves the frame, so holding a sign never repeats it every frame.
    """

    def __init__(self, hold_seconds=1.5):
        self.hold_seconds = hold_seconds
        self._current = None
        self._since = 0.0
        self._fired = False

    def update(self, stable_gesture, now):
        """Feed the current stable gesture (or None). Returns a gesture to add, or None."""
        if stable_gesture != self._current:
            self._current = stable_gesture
            self._since = now
            self._fired = False
            return None
        if stable_gesture is None or self._fired:
            return None
        if now - self._since >= self.hold_seconds:
            self._fired = True
            return stable_gesture
        return None

    def progress(self, now):
        """0..1 progress towards the next auto-add (for a progress bar)."""
        if self._current is None or self._fired:
            return 0.0
        return min(1.0, (now - self._since) / self.hold_seconds)

    def mark_used(self):
        """Call after the current gesture was added manually so the hold doesn't add it twice."""
        self._fired = True

    def reset(self):
        self._current = None
        self._fired = False
