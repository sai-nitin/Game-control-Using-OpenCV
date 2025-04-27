import cv2
import numpy as np
import torch
import mediapipe as mp
import pyautogui
import joblib
import time
from model import PoseClassifier  # Your model class

# === Load Model & Utils ===
model = PoseClassifier(input_size=16, num_classes=5)
model.load_state_dict(torch.load('pose_model.pt'))
model.eval()

scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# === MediaPipe Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
KEYPOINTS = [11, 12, 13, 14]  # Upper torso: shoulders and elbows

# === Key Mapping (no 'none')
key_map = {
    'left': 'left',
    'right': 'right',
    'jump': 'up',
    'down': 'down'
}

# === Control Settings ===
cooldown = 1.5  # seconds
last_action_time = 0
prediction_history = []
required_consistency = 3
frame_count = 0
predict_every = 2

# === Start Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time control. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_count += 1
        if frame_count % predict_every == 0:
            try:
                landmarks = result.pose_landmarks.landmark
                row = []
                for idx in KEYPOINTS:
                    lm = landmarks[idx]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                features_scaled = scaler.transform([row])
                input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

                # === Predict Action ===
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_idx = torch.argmax(output, dim=1).item()
                    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

                if predicted_label == "none":
                    prediction_history.clear()
                    continue

                prediction_history.append(predicted_label)
                if len(prediction_history) > required_consistency:
                    prediction_history.pop(0)

                if prediction_history.count(predicted_label) == required_consistency:
                    now = time.time()
                    if now - last_action_time > cooldown:
                        pyautogui.press(key_map[predicted_label])
                        last_action_time = now
                        print(f"Pressed: {key_map[predicted_label]}")
                        prediction_history.clear()

                # === Show prediction on screen
                cv2.putText(frame, f'Action: {predicted_label.upper()}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error: {e}")

    cv2.imshow("Pose Game Control", frame)
    if cv2.waitKey(10) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
