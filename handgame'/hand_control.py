import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cooldown = 1.0
last_action_time = time.time()
current_action = "None"

# Check if a hand is in a fist (all 4 fingers folded)
def is_fist(hand_landmarks):
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            return False
    return True

# Check if a hand is open (at least 3 fingers extended)
def is_open(hand_landmarks):
    tip_ids = [8, 12, 16, 20]
    extended = 0
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            extended += 1
    return extended >= 3

print("Raise open hand to move left/right, closed hand for up/down. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    action = "None"
    current_time = time.time()

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_open(hand_landmarks):
                if hand_label == "Right":
                    action = "Right"
                elif hand_label == "Left":
                    action = "Left"
            elif is_fist(hand_landmarks):
                if hand_label == "Right":
                    action = "Up"
                elif hand_label == "Left":
                    action = "Down"

    if action != "None" and (current_time - last_action_time > cooldown):
        pyautogui.press(action.lower())
        current_action = action
        last_action_time = current_time
    elif action == "None":
        current_action = "None"

    cv2.putText(frame, f'Action: {current_action}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Strict Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
