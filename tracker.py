"""Webcam hand tracker and open/closed finger counter -- one Python file.

PYCHARM QUICK START
  1. Create a project with Python 3.11 or 3.12 (64-bit).
  2. In that project's Terminal, run:
         python -m pip install mediapipe opencv-contrib-python numpy
  3. Open this file, right-click it, and choose Run.

The first run downloads Google's ~8 MB hand model to your user cache.
After that, hand tracking runs locally without an internet connection.
Webcam frames are never uploaded or recorded by this program.

CONTROLS (click the camera window first)
  Q / Esc : quit       M : mirror on/off       R : reset smoothing

If you use an external webcam, change CAMERA_INDEX below from 0 to 1.
Keep the entire hand visible, with good light and space between hands.
Open/closed states are estimates: partly bent or hidden fingers can be wrong.
The thresholds below can be adjusted for your hand and camera.

API reference:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
"""

from __future__ import annotations

import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except (ImportError, OSError) as exc:
    raise SystemExit(
        f"Could not load a required package: {exc}\n\n"
        "In your PyCharm project's Terminal, run:\n"
        "  python -m pip install mediapipe opencv-contrib-python numpy\n\n"
        "Use the same Python interpreter for installation and Run.\n"
        "Name this file hand_finger_tracker.py, not cv2.py or mediapipe.py."
    ) from exc


# ------------------------- Easy-to-change settings -------------------------
CAMERA_INDEX = 0  # Your laptop's built-in webcam.
MIRROR = True
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
MAX_HANDS = 2
SMOOTHING_FRAMES = 5  # A short majority vote reduces flickering.

# Angles are in degrees; a straight joint is close to 180 degrees.
# Allow natural curvature and noise in estimated 3D finger depth.
FINGER_PIP_ANGLE = 140
FINGER_DIP_ANGLE = 120
THUMB_MCP_ANGLE = 140
THUMB_IP_ANGLE = 150
THUMB_MIN_SPREAD = 0.55  # Thumb-tip to index-base distance / palm width.

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW_NAME = "Hand & Finger Tracker"
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                 (13, 14, 15, 16), (17, 18, 19, 20))
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)
# OpenCV colors are BGR, not RGB.
GREEN = (110, 224, 106)
ORANGE = (80, 154, 255)
WHITE = (240, 243, 248)
MUTED = (160, 172, 187)
BACKGROUND = (23, 28, 37)


def ensure_model() -> Path:
    """Download once, validate the bundle, then keep a local cached copy."""
    root = Path(os.environ.get("LOCALAPPDATA") or
                os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    path = root / "HandFingerTracker" / "hand_landmarker.task"

    def valid(candidate: Path) -> bool:
        try:
            with zipfile.ZipFile(candidate) as bundle:
                return {"hand_detector.tflite", "hand_landmarks_detector.tflite"}.issubset(
                    bundle.namelist()
                ) and bundle.testzip() is None
        except (OSError, zipfile.BadZipFile):
            return False

    if valid(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    print("First run: downloading Google's hand model. Please wait...")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "HandFingerTracker/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
            while chunk := response.read(256 * 1024):
                output.write(chunk)
        if not valid(temporary):
            raise RuntimeError("The model download is incomplete or invalid.")
        temporary.replace(path)
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not download the hand model: {exc}\n"
            "Connect to the internet and run again. You can also download:\n"
            f"{MODEL_URL}\nThen save it as:\n{path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Model ready: {path}")
    return path


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC, using 3D landmarks rather than screen height."""
    first, second = a - b, c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-10:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def classify_fingers(points: np.ndarray) -> list[bool]:
    """Estimate extension in thumb/index/middle/ring/pinky order.

    World landmarks share a 3D metric coordinate system. Angles and distance
    ratios tolerate scale changes and rotation better than tip.y < joint.y.
    These geometric thresholds are heuristics, not calibrated probabilities.
    """
    palm_width = max(float(np.linalg.norm(points[5] - points[17])), 1e-8)
    thumb_open = (
        joint_angle(points[1], points[2], points[3]) > THUMB_MCP_ANGLE
        and joint_angle(points[2], points[3], points[4]) > THUMB_IP_ANGLE
        and np.linalg.norm(points[4] - points[5]) / palm_width > THUMB_MIN_SPREAD
    )
    states = [bool(thumb_open)]
    for mcp, pip, dip, tip in FINGER_CHAINS[1:]:
        straight_pip = joint_angle(points[mcp], points[pip], points[dip]) > FINGER_PIP_ANGLE
        straight_dip = joint_angle(points[pip], points[dip], points[tip]) > FINGER_DIP_ANGLE
        # A finger folded across the palm may still have straight end joints.
        reaches_outward = np.linalg.norm(points[tip] - points[0]) > (
            1.03 * np.linalg.norm(points[pip] - points[0])
        )
        states.append(bool(straight_pip and straight_dip and reaches_outward))
    return states


class StateSmoother:
    """Match nearby wrists across frames and smooth each hand independently."""

    def __init__(self) -> None:
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, detections: list[dict], now: float) -> list[dict]:
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track["last_seen"] < 0.35}
        # Greedy nearest-neighbor matching is sufficient for at most two hands.
        pairs = sorted(
            (float(np.linalg.norm(hand["wrist"] - track["wrist"])), index, key)
            for index, hand in enumerate(detections)
            for key, track in self.tracks.items()
        )
        matches, used = {}, set()
        for distance, index, key in pairs:
            if distance < 0.25 and index not in matches and key not in used:
                matches[index] = key
                used.add(key)
        for index, hand in enumerate(detections):
            key = matches.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = {"history": deque(maxlen=SMOOTHING_FRAMES)}
            track = self.tracks[key]
            track.update(wrist=hand["wrist"], last_seen=now)
            track["history"].append(hand["states"])
            votes = np.mean(track["history"], axis=0)
            previous = track.get("states", hand["states"])
            # Preserve the last state on an exact tie.
            track["states"] = [bool(v > 0.5) if v != 0.5 else previous[i]
                               for i, v in enumerate(votes)]
            hand["states"] = track["states"].copy()
            hand["id"] = key
        return detections


def open_camera():
    """Prefer DirectShow on Windows; try the normal backend if needed."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        camera = cv2.VideoCapture(CAMERA_INDEX, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, 30)
            success, frame = camera.read()
            if success and frame is not None:
                return camera, frame
        camera.release()
    raise RuntimeError(
        f"Cannot read webcam {CAMERA_INDEX}.\n"
        "Close other apps using your camera, enable camera access for desktop apps\n"
        "in Windows Settings > Privacy & security > Camera, then run again.\n"
        "For an external webcam, try CAMERA_INDEX = 1 at the top of this file."
    )


def write_text(canvas, text, position, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hand(frame: np.ndarray, hand: dict) -> None:
    height, width = frame.shape[:2]
    coordinates = [(int(mark.x * width), int(mark.y * height))
                   for mark in hand["landmarks"]]
    for start, end in CONNECTIONS:
        cv2.line(frame, coordinates[start], coordinates[end], GREEN, 2, cv2.LINE_AA)
    for index, point in enumerate(coordinates):
        cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)
    for finger, chain in enumerate(FINGER_CHAINS):
        color = GREEN if hand["states"][finger] else ORANGE
        cv2.circle(frame, coordinates[chain[-1]], 8, color, -1, cv2.LINE_AA)
    x, y = coordinates[0]
    x = max(8, min(x - 35, width - 190))
    y = max(28, min(y + 30, height - 14))
    label = f'{hand["label"]}: {sum(hand["states"])} open'
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    write_text(frame, label, (x, y), color=GREEN, thickness=2)


def make_display(frame: np.ndarray, hands: list[dict], fps: float, mirror: bool) -> np.ndarray:
    """Place the camera beside a readable per-finger information panel."""
    # Keep the display manageable even if the camera ignores the size request.
    height, width = frame.shape[:2]
    scale = min(960 / width, 720 / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    height, width = frame.shape[:2]
    panel_width, header = 310, 66
    display_height = max(height, 600)
    canvas = np.full((display_height + header, width + panel_width, 3), BACKGROUND, dtype=np.uint8)
    canvas[header:header + height, :width] = frame
    write_text(canvas, "HAND & FINGER TRACKER", (20, 29), 0.75, thickness=2)
    write_text(canvas, "Green = open    Orange = closed", (20, 51), 0.46, MUTED)
    left = width + 18
    total_open = sum(sum(hand["states"]) for hand in hands)
    total_closed = 5 * len(hands) - total_open
    write_text(canvas, f"HANDS {len(hands)}    FPS {fps:.0f}", (left, 31), 0.56, MUTED)
    write_text(canvas, f"OPEN: {total_open}", (left, 82), 1.05, GREEN, 2)
    write_text(canvas, f"CLOSED: {total_closed}", (left, 117), 0.72, ORANGE, 2)

    for slot, hand in enumerate(sorted(hands, key=lambda item: item["id"])):
        top = 163 + slot * 205
        opened = sum(hand["states"])
        write_text(canvas, f'{hand["label"]} hand  |  {opened}/5 open',
                   (left, top), 0.58, WHITE, 2)
        for row, (name, is_open) in enumerate(zip(FINGER_NAMES, hand["states"])):
            y = top + 31 + row * 27
            write_text(canvas, name, (left, y), 0.54, MUTED)
            write_text(canvas, "OPEN" if is_open else "CLOSED", (left + 132, y),
                       0.54, GREEN if is_open else ORANGE, 2)

    if not hands:
        write_text(canvas, "Show one or both hands", (left, 179), 0.56)
        write_text(canvas, "Keep all fingers in view.", (left, 208), 0.5, MUTED)
        write_text(canvas, "No hand = no finger reading.", (left, 237), 0.47, MUTED)
    footer = canvas.shape[0] - 49
    write_text(canvas, "Q / Esc: quit    R: reset", (left, footer), 0.48, MUTED)
    write_text(canvas, f'M: mirror ({"ON" if mirror else "OFF"})',
               (left, footer + 25), 0.48, MUTED)
    write_text(canvas, "Use good light. Partly bent / hidden fingers may be misread.",
               (18, canvas.shape[0] - 17), 0.46, MUTED)
    return canvas


def main() -> None:
    model_path = ensure_model()
    options = vision.HandLandmarkerOptions(
        # Bytes also avoid model-path issues with non-ASCII Windows usernames.
        base_options=python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    camera = None
    smoother = StateSmoother()
    mirror = MIRROR
    last_timestamp = -1
    previous_time = time.perf_counter()
    fps = 0.0
    print("Starting webcam. Click the video window; press Q or Esc to quit.")
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            camera, frame = open_camera()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1200, 640)
            while True:
                now = time.perf_counter()
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                # VIDEO mode permits synchronous webcam inference. Strictly
                # increasing timestamps are required even on very fast frames.
                timestamp_ms = max(last_timestamp + 1, int(now * 1000))
                last_timestamp = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                hands = []
                for index, landmarks in enumerate(result.hand_landmarks):
                    world = result.hand_world_landmarks[index]
                    points = np.array([(point.x, point.y, point.z) for point in world])
                    label = result.handedness[index][0].category_name
                    # MediaPipe handedness labels assume a mirrored/selfie view.
                    if not mirror:
                        label = {"Left": "Right", "Right": "Left"}.get(label, label)
                    hands.append({
                        "landmarks": landmarks,
                        "wrist": np.array([landmarks[0].x, landmarks[0].y]),
                        "label": label,
                        "states": classify_fingers(points),
                    })
                hands = smoother.update(hands, now)
                for hand in hands:
                    draw_hand(frame, hand)
                finished = time.perf_counter()
                instantaneous_fps = 1.0 / max(finished - previous_time, 1e-6)
                fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
                previous_time = finished
                cv2.imshow(WINDOW_NAME, make_display(frame, hands, fps, mirror))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("m"), ord("M")):
                    mirror = not mirror
                    smoother = StateSmoother()
                elif key in (ord("r"), ord("R")):
                    smoother = StateSmoother()
                success, frame = camera.read()
                if not success or frame is None:
                    raise RuntimeError("Webcam stopped returning frames. Reconnect it and run again.")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"\nCould not start or continue hand tracking:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
