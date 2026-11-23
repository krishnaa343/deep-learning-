import sys
import os
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import tensorflow as tf

# Add app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from modelutil import load_model
from utils import decode_prediction

TARGET_FRAMES = 75
TARGET_H = 46
TARGET_W = 140

def main():
    print("=" * 65)
    print("  Silent Speech AI - Live Camera Lip-Reading on the Spot")
    print("=" * 65)
    print("Loading 3D-CNN + BiLSTM Deep Learning Model into GPU/RAM...")
    model = load_model()
    print("✓ Model successfully loaded and ready!")

    # Initialize Face Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load cascade from {cascade_path}")
        return

    # Open Camera
    print("Connecting to live camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n✗ Error: Could not access webcam. Please ensure a camera is connected and permitted.")
        input("Press Enter to exit...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State variables
    state = "IDLE"  # "IDLE", "RECORDING", "CONTINUOUS"
    recorded_lip_frames = []
    continuous_buffer = deque(maxlen=TARGET_FRAMES)
    last_prediction = ""
    status_msg = "READY: Speak clearly or press [SPACE] to record"
    status_color = (0, 220, 0)
    smoothed_mouth_box = None
    alpha = 0.35  # Smoothing factor to eliminate bounding box jitter
    
    # Motion detection variables for auto-triggering speech
    prev_lip_gray = None
    motion_history = deque(maxlen=5)
    auto_trigger_cooldown = 0

    window_name = "Silent Speech AI - Live Lip Reading on the Spot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[Controls]:")
    print("  • Simply speak with your lips (Auto-detects speech)")
    print("  • Or press [SPACE] to manually record 75 frames (2.5s)")
    print("  • Press [C] to toggle Continuous rolling mode")
    print("  • Press [Q] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame from camera.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        lip_for_model = None
        lip_motion = 0.0

        if len(faces) > 0:
            # Pick largest face
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            
            # Geometric mouth region calculation
            raw_my1 = fy + int(fh * 0.63)
            raw_my2 = fy + int(fh * 0.94)
            raw_mx1 = fx + int(fw * 0.20)
            raw_mx2 = fx + int(fw * 0.80)

            # Ensure bounds
            raw_my1 = max(0, min(cam_h - 1, raw_my1))
            raw_my2 = max(0, min(cam_h, raw_my2))
            raw_mx1 = max(0, min(cam_w - 1, raw_mx1))
            raw_mx2 = max(0, min(cam_w, raw_mx2))

            # Exponential smoothing to prevent bounding box jitter
            if smoothed_mouth_box is None:
                smoothed_mouth_box = [raw_mx1, raw_my1, raw_mx2, raw_my2]
            else:
                smoothed_mouth_box[0] = int(smoothed_mouth_box[0] * (1 - alpha) + raw_mx1 * alpha)
                smoothed_mouth_box[1] = int(smoothed_mouth_box[1] * (1 - alpha) + raw_my1 * alpha)
                smoothed_mouth_box[2] = int(smoothed_mouth_box[2] * (1 - alpha) + raw_mx2 * alpha)
                smoothed_mouth_box[3] = int(smoothed_mouth_box[3] * (1 - alpha) + raw_my2 * alpha)

            mx1, my1, mx2, my2 = smoothed_mouth_box

            # Draw Face Outline (Soft Green)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
            cv2.putText(frame, "Face Tracked", (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # Draw Lip Focus Box (Bright Cyan)
            if my2 > my1 + 10 and mx2 > mx1 + 10:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 220, 0), 2)
                cv2.putText(frame, "LIPS (Model Focus)", (mx1, my1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

                lip_crop = gray[my1:my2, mx1:mx2]
                if lip_crop.size > 0:
                    # Resize to (140, 46)
                    resized_lip = cv2.resize(lip_crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                    lip_for_model = np.expand_dims(resized_lip, axis=-1)

                    # Compute mouth motion for speech detection
                    if prev_lip_gray is not None:
                        diff = cv2.absdiff(resized_lip, prev_lip_gray)
                        lip_motion = float(np.mean(diff))
                        motion_history.append(lip_motion)
                    prev_lip_gray = resized_lip.copy()

        # Check for automatic speech trigger based on sustained lip motion
        if auto_trigger_cooldown > 0:
            auto_trigger_cooldown -= 1

        avg_motion = np.mean(motion_history) if motion_history else 0.0
        if state == "IDLE" and auto_trigger_cooldown == 0 and len(motion_history) >= 3 and avg_motion > 12.0:
            # Lip movement detected! Auto-start recording
            state = "RECORDING"
            recorded_lip_frames = []
            status_msg = "🎙️ Mouth movement detected! Recording lips..."
            status_color = (0, 0, 255)

        # PiP Preview of Lip ROI in top right corner
        if lip_for_model is not None:
            pip_w, pip_h = 140, 46
            pip_view = cv2.cvtColor(np.squeeze(lip_for_model), cv2.COLOR_GRAY2BGR)
            pip_x = cam_w - pip_w - 15
            pip_y = 55
            cv2.rectangle(frame, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (255, 220, 0), 2)
            frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_view
            cv2.putText(frame, "Lip ROI Feed", (pip_x, pip_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Top Header Overlay
        cv2.rectangle(frame, (0, 0), (cam_w, 45), (25, 25, 25), -1)
        cv2.putText(frame, status_msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Handle Recording & Prediction
        if state == "RECORDING":
            if lip_for_model is not None:
                recorded_lip_frames.append(lip_for_model)

            progress = len(recorded_lip_frames)
            status_msg = f"● READING LIPS: [{progress}/{TARGET_FRAMES} frames]"
            status_color = (0, 0, 255)

            # Draw progress bar on camera feed
            bar_w = int((progress / TARGET_FRAMES) * (cam_w - 30))
            cv2.rectangle(frame, (15, 42), (15 + bar_w, 47), (0, 0, 255), -1)

            if progress >= TARGET_FRAMES:
                status_msg = "🧠 CONVERTING TO TEXT ON THE SPOT..."
                status_color = (0, 165, 255)

                # Prepare tensor: (75, 46, 140, 1)
                frames_arr = np.array(recorded_lip_frames, dtype=np.float32)
                mean = np.mean(frames_arr)
                std = np.std(frames_arr)
                norm_frames = (frames_arr - mean) / (std + 1e-6)

                # Predict with model
                model_input = tf.expand_dims(norm_frames, axis=0)
                yhat = model.predict(model_input, verbose=0)
                ctc_decoded = tf.keras.backend.ctc_decode(yhat, [TARGET_FRAMES], greedy=True)[0][0].numpy()
                prediction = decode_prediction(ctc_decoded)
                
                if prediction.strip():
                    last_prediction = prediction
                else:
                    last_prediction = "(Uncertain / silence)"

                print(f"🎯 Recognized on the spot: '{last_prediction}'")
                recorded_lip_frames = []
                state = "IDLE"
                auto_trigger_cooldown = 20  # Cooldown before auto-triggering again
                status_msg = "READY: Speak clearly or press [SPACE]"
                status_color = (0, 220, 0)

        elif state == "CONTINUOUS":
            if lip_for_model is not None:
                continuous_buffer.append(lip_for_model)

            status_msg = f"⚡ CONTINUOUS ROLLING MODE ({len(continuous_buffer)}/{TARGET_FRAMES})"
            status_color = (0, 180, 255)

            if len(continuous_buffer) == TARGET_FRAMES:
                frames_arr = np.array(continuous_buffer, dtype=np.float32)
                mean = np.mean(frames_arr)
                std = np.std(frames_arr)
                norm_frames = (frames_arr - mean) / (std + 1e-6)
                model_input = tf.expand_dims(norm_frames, axis=0)
                yhat = model.predict(model_input, verbose=0)
                ctc_decoded = tf.keras.backend.ctc_decode(yhat, [TARGET_FRAMES], greedy=True)[0][0].numpy()
                pred = decode_prediction(ctc_decoded)
                if pred.strip():
                    last_prediction = pred

        # =========================================================================
        # CREATE DEDICATED BOTTOM RESULT PANEL ("TELL ON THE SPOT BELOW THERE")
        # =========================================================================
        panel_h = 130
        bottom_panel = np.zeros((panel_h, cam_w, 3), dtype=np.uint8)
        bottom_panel[:] = (18, 18, 18)  # Sleek dark background

        # Top border line of panel
        cv2.line(bottom_panel, (0, 0), (cam_w, 0), (50, 50, 50), 2)

        # Header in bottom panel
        cv2.putText(bottom_panel, "🎯 RECOGNIZED FROM YOUR LIPS ON THE SPOT:", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Big bold text result
        disp_text = f'"{last_prediction}"' if last_prediction else '"(Listening for lip movement...)"'
        text_color = (0, 255, 100) if last_prediction else (120, 120, 120)
        cv2.putText(bottom_panel, disp_text, (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_color, 2, cv2.LINE_AA)

        # Bottom shortcut hints
        cv2.putText(bottom_panel, "[SPACE] Manual Record  |  [C] Continuous Mode  |  [Q] Exit", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Combine camera frame and bottom panel vertically
        combined_display = np.vstack([frame, bottom_panel])

        cv2.imshow(window_name, combined_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:  # Esc or Q
            break
        elif key == 32:  # SPACE bar
            if state != "RECORDING":
                state = "RECORDING"
                recorded_lip_frames = []
                last_prediction = ""
                status_msg = "● RECORDING: Speak your sentence now!"
                status_color = (0, 0, 255)
        elif key == ord('c') or key == ord('C'):
            if state == "CONTINUOUS":
                state = "IDLE"
                status_msg = "READY: Speak clearly or press [SPACE]"
                status_color = (0, 220, 0)
            else:
                state = "CONTINUOUS"
                continuous_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. App closed.")

if __name__ == "__main__":
    main()
