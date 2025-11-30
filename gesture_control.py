import os
import sys
import time
import platform
import subprocess
from collections import deque
from math import hypot

import cv2
import numpy as np
import pyautogui

# Linux volume/brightness/media helpers
def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except:
        return False

# Volume
def set_volume(pct):
    pct = int(max(0, min(100, pct)))
    run_cmd(f"pactl set-sink-volume @DEFAULT_SINK@ {pct}%")

# Brightness
def set_brightness(pct):
    pct = int(max(0, min(100, pct)))
    run_cmd(f"brightnessctl set {pct}%")

# Media
def play_pause():
    run_cmd("playerctl play-pause")

def next_track():
    run_cmd("playerctl next")

def prev_track():
    run_cmd("playerctl previous")


# ───────────────────────────────
#     MEDIAPIPE HAND TRACKING
# ───────────────────────────────
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def dist(p1, p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])


def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    ups = []
    for t, p in zip(tips, pips):
        ups.append(landmarks[t][1] < landmarks[p][1])
    return ups


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected")
        sys.exit()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        prev_area = None
        x_history = deque(maxlen=8)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w = frame.shape[:2]

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = []

                for p in hand.landmark:
                    lm.append((int(p.x * w), int(p.y * h)))

                ups = fingers_up(lm)
                cnt = sum(ups)

                # Draw hand
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS)

                # Gesture 1 → Fist → Play/Pause
                if cnt == 0:
                    play_pause()
                    cv2.putText(frame, "Play/Pause", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    time.sleep(0.7)

                # Gesture 2 → Palm + swipe → next/prev track
                x_history.append(lm[8][0])

                if cnt >= 4 and len(x_history) == x_history.maxlen:
                    dx = x_history[-1] - x_history[0]

                    if dx > 180:
                        next_track()
                        cv2.putText(frame, "Next Track", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        time.sleep(0.7)
                        x_history.clear()

                    elif dx < -180:
                        prev_track()
                        cv2.putText(frame, "Previous Track", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        time.sleep(0.7)
                        x_history.clear()

                # Gesture 3 → Pinch (Thumb + Index) → Volume Control
                thumb = lm[4]
                index = lm[8]
                distance = dist(thumb, index)

                vol = int(np.interp(distance, [20, 250], [0, 100]))
                set_volume(vol)

                cv2.putText(frame, f"Volume: {vol}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Gesture 4 → Palm up/down → Brightness
                if cnt >= 4:
                    brightness = int(np.interp(lm[0][1], [h*0.8, h*0.2], [0, 100]))
                    set_brightness(brightness)
                    cv2.putText(frame, f"Brightness: {brightness}", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)

            cv2.imshow("Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

