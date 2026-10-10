"""
Webcam + MediaPipe hand tracking for VisionSpeak's Sign to Text feature.

SignCamera runs capture, hand-landmark detection and gesture classification
on a background thread. The Tkinter UI never touches the camera directly;
it polls `latest()` from the main thread (via `window.after`) and draws the
most recent frame. Everything runs locally, no internet connection needed.

Uses the MediaPipe Tasks API (HandLandmarker). The older `mp.solutions.hands`
API was removed from recent MediaPipe releases.
"""

import os
import sys
import threading
import time

import gestures

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# Give up after this many failed frame reads in a row (camera unplugged, etc.).
MAX_READ_FAILURES = 30


class SignCameraError(Exception):
    """A problem the user can fix (missing package, model or camera)."""


def check_dependencies(model_path=MODEL_PATH):
    """Raise SignCameraError with install instructions if something is missing."""
    missing = []
    for module, package in (("cv2", "opencv-contrib-python"), ("mediapipe", "mediapipe"),
                            ("PIL", "Pillow"), ("numpy", "numpy")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise SignCameraError(
            "Missing Python packages: " + ", ".join(missing) +
            "\n\nInstall them with:\n    pip install -r requirements.txt")
    if not os.path.isfile(model_path):
        raise SignCameraError(
            f"Hand-landmark model not found:\n{model_path}\n\n"
            f"Download it once (about 7.5 MB) from:\n{MODEL_URL}\n"
            "and save it in the 'models' folder.")


class SignCamera:
    def __init__(self, camera_index=0, model_path=MODEL_PATH):
        self.camera_index = camera_index
        self.model_path = model_path
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the worker thread. Raises SignCameraError if dependencies are missing."""
        if self.running:
            return
        check_dependencies(self.model_path)
        self._stop.clear()
        self._error = None
        self._latest = None
        self._thread = threading.Thread(target=self._run, name="SignCamera", daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        """Ask the worker to stop and wait for it to release the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def latest(self):
        """
        Thread-safe snapshot of the newest result, or None before the first frame:
        {"frame": RGB numpy array with landmarks drawn, "hand": bool,
         "gesture": str, "confidence": float, "stable": str or None, "fps": float}
        """
        with self._lock:
            return self._latest

    @property
    def error(self):
        """Error message if the worker stopped because of a problem, else None."""
        return self._error

    # ------------------------------------------------------------ worker thread
    def _run(self):
        capture = None
        landmarker = None
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            capture = self._open_camera(cv2)
            landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            ))
            connections = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
            smoother = gestures.GestureSmoother()

            failures = 0
            last_ts = 0
            fps = 0.0
            prev_time = time.monotonic()
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        raise SignCameraError("The webcam stopped sending frames. Is it still connected?")
                    time.sleep(0.05)
                    continue
                failures = 0

                frame = cv2.flip(frame, 1)  # mirror view feels natural; rules are mirror-safe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                # VIDEO mode needs strictly increasing timestamps.
                ts = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                hand = bool(result.hand_landmarks)
                gesture, confidence = gestures.UNKNOWN, 0.0
                if hand:
                    # Pixel units so x and y share a scale; z uses the same scale as x.
                    points = [(p.x * w, p.y * h, p.z * w) for p in result.hand_landmarks[0]]
                    gesture, confidence = gestures.classify(points)
                    _draw_hand(cv2, rgb, points, connections)
                stable = smoother.update(gesture if hand else None, confidence)

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-3))
                prev_time = now

                with self._lock:
                    self._latest = {"frame": rgb, "hand": hand, "gesture": gesture,
                                    "confidence": confidence, "stable": stable, "fps": fps}
        except SignCameraError as exc:
            self._error = str(exc)
        except Exception as exc:  # report anything unexpected instead of dying silently
            self._error = f"Camera error: {exc}"
        finally:
            if landmarker is not None:
                landmarker.close()
            if capture is not None:
                capture.release()

    def _open_camera(self, cv2):
        # DirectShow opens much faster than the default backend on Windows.
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, _ = capture.read()
                if ok:
                    return capture
            capture.release()
        raise SignCameraError(
            "Could not open the webcam.\n\n"
            "- Check that a camera is connected.\n"
            "- Close other apps that may be using it (Zoom, Teams, Camera).\n"
            "- Windows: Settings > Privacy & security > Camera > allow desktop apps.")


def _draw_hand(cv2, image, points, connections):
    pts = [(int(x), int(y)) for x, y, _ in points]
    for a, b in connections:
        cv2.line(image, pts[a], pts[b], (124, 92, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        color = (255, 255, 255) if i in (4, 8, 12, 16, 20) else (56, 189, 248)
        cv2.circle(image, p, 4, color, -1, cv2.LINE_AA)
