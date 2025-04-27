import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import threading
import tkinter as tk

# Config
DATA_PATH = 'pose_data_v2.csv'
KEYPOINTS = [11, 12, 13, 14]  # L/R shoulder, L/R elbow
LABELS = ['left', 'right', 'jump', 'down', 'none']

# Setup CSV
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{i}_{axis}' for i in range(4) for axis in ['x', 'y', 'z', 'v']]
        writer.writerow(header)

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# App State
save_next = False
current_label = "none"
current_features = None
running = True

# Save next frame
def save_pose(label):
    global save_next, current_label
    save_next = True
    current_label = label

# Stop app
def quit_app():
    global running
    running = False
    root.quit()

# Webcam Thread
def start_camera():
    global save_next, current_features, running

    cap = cv2.VideoCapture(0)
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        current_features = None
        row = []

        if result.pose_landmarks:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                for idx in KEYPOINTS:
                    lm = result.pose_landmarks.landmark[idx]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                current_features = row
            except:
                pass

        if save_next and current_features is not None:
            with open(DATA_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + current_features)
            print(f"✅ Saved: {current_label}")
            save_next = False

        cv2.imshow("Pose Collector", frame)
        if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Pose Data Collector")

tk.Label(root, text="Click a button to record pose").pack()

for lbl in LABELS:
    tk.Button(root, text=lbl.upper(), width=20, height=2, command=lambda l=lbl: save_pose(l)).pack(pady=2)

tk.Button(root, text="❌ Quit", width=20, height=2, bg="red", fg="white", command=quit_app).pack(pady=5)

# Start camera thread
threading.Thread(target=start_camera).start()

# Start GUI
root.mainloop()
