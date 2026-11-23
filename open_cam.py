import sys
import os
import json
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "open_vsr"
MODEL_PATH = MODEL_DIR / "lipread_ctc.int8.onnx"
TOKENS_PATH = MODEL_DIR / "tokens.json"

TARGET_SIZE = 88
MAX_FRAMES = 400
MIN_FRAMES = 20
SILENCE_THRESHOLD = 35


class OpenVocabularyLipReader:
    def __init__(self):
        print("Loading Auto-AVSR Open-Vocabulary Model...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(MODEL_PATH), opts)
        self.input_name = self.session.get_inputs()[0].name
        with open(TOKENS_PATH, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        print(f"  Model loaded. Vocabulary: {len(self.tokens)} subword pieces (open English).")

    def predict(self, crops):
        if len(crops) < 8:
            return ""
        arr = np.array(crops, dtype=np.float32)
        x = (arr / 255.0 - 0.421) / 0.165
        log_probs = self.session.run(["log_probs"], {self.input_name: x[None, None]})[0]
        ids, prev = [], None
        eos = len(self.tokens) - 1
        for i in log_probs.argmax(-1):
            if i != prev and i != 0 and i != eos:
                ids.append(int(i))
            prev = i
        if not ids:
            return ""
        return "".join(self.tokens[i] for i in ids).replace("\u2581", " ").strip()


def main():
    print("=" * 60)
    print("  Auto-AVSR  Open-Vocabulary Live Lip Reader")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run the setup first.")
        return

    reader = OpenVocabularyLipReader()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        input("Press Enter to exit...")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- state ---
    state = "IDLE"
    manual = False
    crops = []
    last_text = ""
    smooth = None
    a = 0.35
    prev_mouth = None
    motion_buf = deque(maxlen=7)
    cooldown = 0
    silent_count = 0

    win = "Auto-AVSR  Live Lip Reader"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print()
    print("Controls:")
    print("  [SPACE]  Start recording  /  Stop and transcribe")
    print("  [C]      Toggle continuous mode")
    print("  [Q]      Quit")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
        crop88 = None
        motion = 0.0

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx = fx + fw // 2
            cy = fy + int(fh * 0.77)
            half = int(fh * 0.24)
            rx1, ry1 = max(0, cx - half), max(0, cy - half)
            rx2, ry2 = min(w, cx + half), min(h, cy + half)

            if smooth is None:
                smooth = [rx1, ry1, rx2, ry2]
            else:
                for k, v in enumerate([rx1, ry1, rx2, ry2]):
                    smooth[k] = int(smooth[k] * (1 - a) + v * a)

            mx1, my1, mx2, my2 = smooth

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
            if my2 - my1 > 10 and mx2 - mx1 > 10:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 220, 0), 2)
                cv2.putText(frame, "LIPS", (mx1, my1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)
                roi = gray[my1:my2, mx1:mx2]
                if roi.size > 0:
                    crop88 = cv2.resize(roi, (TARGET_SIZE, TARGET_SIZE))
                    if prev_mouth is not None:
                        motion = float(np.mean(cv2.absdiff(crop88, prev_mouth)))
                        motion_buf.append(motion)
                    prev_mouth = crop88.copy()

        if cooldown > 0:
            cooldown -= 1
        avg_m = float(np.mean(motion_buf)) if motion_buf else 0.0

        # auto-trigger
        if state == "IDLE" and cooldown == 0 and len(motion_buf) >= 4 and avg_m > 8.5:
            state = "RECORDING"
            manual = False
            crops = []
            silent_count = 0

        # pip preview
        if crop88 is not None:
            pip = cv2.cvtColor(crop88, cv2.COLOR_GRAY2BGR)
            px, py = w - TARGET_SIZE - 12, 52
            if py + TARGET_SIZE < h and px + TARGET_SIZE < w:
                cv2.rectangle(frame, (px - 2, py - 2),
                              (px + TARGET_SIZE + 2, py + TARGET_SIZE + 2),
                              (255, 220, 0), 2)
                frame[py:py + TARGET_SIZE, px:px + TARGET_SIZE] = pip

        # header
        cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
        if state == "RECORDING":
            secs = round(len(crops) / 25.0, 1)
            if manual:
                hdr = f"REC  {secs}s  -  Press [SPACE] when done speaking"
            else:
                hdr = f"Listening  {secs}s  -  Keep speaking..."
            cv2.putText(frame, hdr, (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            bar = int(min(len(crops), MAX_FRAMES) / MAX_FRAMES * (w - 28))
            cv2.rectangle(frame, (14, 40), (14 + bar, 44), (0, 0, 255), -1)
        elif state == "CONTINUOUS":
            cv2.putText(frame, "Continuous mode active", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "READY  -  Press [SPACE] to record, or just speak", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)

        # recording logic
        if state == "RECORDING":
            if crop88 is not None:
                crops.append(crop88)
            n = len(crops)

            if avg_m < 4.0 and n >= MIN_FRAMES:
                silent_count += 1
            else:
                silent_count = 0

            do_finish = False
            if manual:
                do_finish = n >= MAX_FRAMES
            else:
                do_finish = silent_count >= SILENCE_THRESHOLD or n >= MAX_FRAMES

            if do_finish:
                last_text = reader.predict(crops) or "(silence)"
                print(f"  >> {last_text}")
                crops = []
                state = "IDLE"
                manual = False
                cooldown = 20

        elif state == "CONTINUOUS":
            if crop88 is not None:
                motion_buf.append(motion)
            # accumulate in fixed 75-frame batches
            if crop88 is not None:
                crops.append(crop88)
            if len(crops) >= 75:
                t = reader.predict(crops)
                if t.strip():
                    last_text = t
                    print(f"  >> {last_text}")
                crops = []

        # bottom subtitle panel
        panel = np.zeros((120, w, 3), dtype=np.uint8)
        panel[:] = (15, 15, 15)
        cv2.line(panel, (0, 0), (w, 0), (60, 60, 60), 2)
        cv2.putText(panel, "RECOGNIZED FROM YOUR LIPS:", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        txt = f'"{last_text}"' if last_text else '"(waiting...)"'
        col = (0, 255, 100) if last_text else (100, 100, 100)
        cv2.putText(panel, txt, (18, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        cv2.putText(panel, "[SPACE] Record / Stop   [C] Continuous   [Q] Quit",
                    (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

        cv2.imshow(win, np.vstack([frame, panel]))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key == 32:  # SPACE
            if state == "RECORDING":
                # stop and transcribe now
                last_text = reader.predict(crops) or "(too short)"
                print(f"  >> {last_text}")
                crops = []
                state = "IDLE"
                manual = False
                cooldown = 15
            else:
                # start manual recording
                state = "RECORDING"
                manual = True
                crops = []
                silent_count = 0
                last_text = ""
                print("  [REC] Speak your sentence, press [SPACE] when done.")

        elif key in (ord("c"), ord("C")):
            if state == "CONTINUOUS":
                state = "IDLE"
                crops = []
            else:
                state = "CONTINUOUS"
                crops = []

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
