from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from counter import SixSevenCounter

APP_DIR = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = APP_DIR / "models" / "hand_landmarker.task"
HIGHSCORE_PATH = APP_DIR / "highscore.json"
WINDOW = "67 Counter"
COUNTDOWN_SECONDS = 3

PALM_IDS = (0, 5, 9, 13, 17)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (120, 230, 120)
RED = (90, 90, 245)
YELLOW = (60, 220, 250)
CYAN = (250, 220, 90)


class State(Enum):
    READY = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    RESULT = auto()


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("First run: downloading the hand model (~8 MB)...")
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"Could not download the model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


def load_highscores() -> dict[str, int]:
    try:
        return json.loads(HIGHSCORE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_highscores(scores: dict[str, int]) -> None:
    try:
        HIGHSCORE_PATH.write_text(json.dumps(scores, indent=2))
    except OSError as exc:
        print(f"Could not save high score: {exc}")


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {index}. Try another one with --camera 1, "
            "and check that no other app is using the webcam."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def create_landmarker(model_path: Path) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def palm_centres(result) -> list[tuple[float, float]]:
    centres = []
    for landmarks in result.hand_landmarks:
        xs = [landmarks[i].x for i in PALM_IDS]
        ys = [landmarks[i].y for i in PALM_IDS]
        centres.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centres


def left_right_heights(centres: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if len(centres) < 2:
        return None, None
    ordered = sorted(centres, key=lambda c: c[0])
    return ordered[0][1], ordered[-1][1]


def put_text(img, text, org, scale=1.0, colour=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, org, font, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def put_text_centred(img, text, centre, scale=1.0, colour=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    put_text(img, text, (int(centre[0] - tw / 2), int(centre[1] + th / 2)), scale, colour, thickness)


def draw_hands(img, result, both_visible: bool) -> None:
    h, w = img.shape[:2]
    colour = GREEN if both_visible else YELLOW
    for landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)
    centres = palm_centres(result)
    pixel = [(int(x * w), int(y * h)) for x, y in sorted(centres)]
    for p in pixel:
        cv2.circle(img, p, 10, CYAN, 2, cv2.LINE_AA)
    if len(pixel) >= 2:
        cv2.line(img, pixel[0], pixel[-1], CYAN, 1, cv2.LINE_AA)


def dim(img, amount: float = 0.45) -> None:
    img[:] = (img * (1 - amount)).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    model_path = ensure_model()
    cap = open_camera(args.camera, args.width, args.height)
    scores = load_highscores()
    key_for_score = str(args.duration)
    best = scores.get(key_for_score, 0)

    counter = SixSevenCounter(threshold=args.threshold)
    state = State.READY
    state_started = time.monotonic()
    round_started = 0.0
    last_beat_time = -10.0
    new_record = False
    last_ts_ms = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    landmarker = create_landmarker(model_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = max(int(now * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)

            centres = palm_centres(result)
            left_y, right_y = left_right_heights(centres)
            both_visible = left_y is not None

            if state is State.COUNTDOWN:
                remaining = COUNTDOWN_SECONDS - (now - state_started)
                if remaining <= 0:
                    state = State.PLAYING
                    counter.reset()
                    round_started = now
            elif state is State.PLAYING:
                if counter.update(left_y, right_y, now):
                    last_beat_time = now
                if now - round_started >= args.duration:
                    state = State.RESULT
                    state_started = now
                    new_record = counter.pairs > best
                    if new_record:
                        best = counter.pairs
                        scores[key_for_score] = best
                        save_highscores(scores)

            if state in (State.READY, State.RESULT):
                dim(frame, 0.35)
            draw_hands(frame, result, both_visible)

            if state is State.READY:
                put_text_centred(frame, "67 COUNTER", (w / 2, h * 0.22), 2.4, YELLOW, 4)
                put_text_centred(frame, "Both palms up. Move your hands up and down,", (w / 2, h * 0.38), 0.9)
                put_text_centred(frame, "one high while the other is low: six... seven...", (w / 2, h * 0.44), 0.9)
                status = "Both hands detected" if both_visible else "Show both hands to the camera"
                put_text_centred(frame, status, (w / 2, h * 0.58), 1.0, GREEN if both_visible else YELLOW)
                put_text_centred(frame, f"Press SPACE to start  ({args.duration}s round)", (w / 2, h * 0.70), 1.1)
                put_text_centred(frame, f"Best: {best}", (w / 2, h * 0.80), 1.0, CYAN)

            elif state is State.COUNTDOWN:
                put_text_centred(frame, "Get ready!", (w / 2, h * 0.25), 1.4)
                put_text_centred(frame, str(math.ceil(max(remaining, 0.01))), (w / 2, h * 0.52), 6.0, YELLOW, 8)

            elif state is State.PLAYING:
                time_left = max(0.0, args.duration - (now - round_started))
                put_text(frame, f"TIME {time_left:4.1f}", (20, 50), 1.2, RED if time_left < 5 else WHITE)
                put_text(frame, f"67s: {counter.pairs}", (20, 100), 1.4, YELLOW, 3)
                put_text(frame, f"Best {best}", (w - 190, 50), 1.0, CYAN)
                age = now - last_beat_time
                if age < 0.3 and counter.beats:
                    scale = 4.0 + 2.5 * (1 - age / 0.3)
                    put_text_centred(frame, counter.last_label, (w / 2, h * 0.5), scale, YELLOW, 8)
                if not both_visible:
                    put_text_centred(frame, "Show both hands!", (w / 2, h - 50), 1.2, RED)

            else:
                put_text_centred(frame, "TIME'S UP!", (w / 2, h * 0.2), 2.2, YELLOW, 4)
                put_text_centred(frame, f"{counter.pairs} x 67", (w / 2, h * 0.43), 3.2, WHITE, 6)
                rate = counter.pairs / args.duration
                put_text_centred(frame, f"{counter.beats} beats  |  {rate:.1f} per second", (w / 2, h * 0.57), 1.0)
                if new_record:
                    put_text_centred(frame, "NEW HIGH SCORE!", (w / 2, h * 0.67), 1.4, GREEN, 3)
                else:
                    put_text_centred(frame, f"Best: {best}", (w / 2, h * 0.67), 1.2, CYAN)
                put_text_centred(frame, "SPACE = play again    Q = quit", (w / 2, h * 0.82), 1.0)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" ") and state in (State.READY, State.RESULT):
                state = State.COUNTDOWN
                state_started = now
            elif key == ord("r"):
                state = State.READY
                state_started = now
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="67 Mediapipe Counter")
    p.add_argument("--duration", type=int, default=30, help="round length in seconds (default 30)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=1280, help="requested capture width")
    p.add_argument("--height", type=int, default=720, help="requested capture height")
    p.add_argument(
        "--threshold", type=float, default=0.06,
        help="how far apart the hands must be, as a fraction of frame height (lower = more sensitive)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
