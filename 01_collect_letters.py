import cv2
import numpy as np
import os
import mediapipe as mp
import string

# All 26 letters A-Z (or customize to ['M', 'N', 'S', 'T'] to re-record specific letters)
LETTERS = list(string.ascii_uppercase)
SAMPLES_PER_LETTER = 100
DATA_PATH = os.path.join('letter_data')
WINDOW_NAME = 'Letter Collector'

for letter in LETTERS:
    os.makedirs(os.path.join(DATA_PATH, letter), exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_hand_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        # 1. Center relative to wrist (landmark 0)
        coords = coords - coords[0]
        
        # 2. Scale normalize so camera distance doesn't affect values
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist
            
        # 3. Add fingertip-to-thumb distances (helps distinguish A/E/M/N/S/T)
        tips = [4, 8, 12, 16, 20]
        thumb_tip = coords[4]
        tip_distances = [np.linalg.norm(coords[t] - thumb_tip) for t in tips[1:]]
        
        return np.concatenate([coords.flatten(), tip_distances])  # 67 features
    return None

def window_closed(win_name):
    return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow(WINDOW_NAME)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    for letter in LETTERS:
        # 1. Preview phase: Show hand landmarks live while you form the letter
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.putText(frame, f'Show "{letter}" & press S to capture (Q or X to quit)', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q') or window_closed(WINDOW_NAME):
                cap.release()
                cv2.destroyAllWindows()
                print("Closed by user.")
                exit()

        # 2. Recording phase: Capture 100 valid hand frames
        count = 0
        while count < SAMPLES_PER_LETTER:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            keypoints = extract_hand_landmarks(results)
            if keypoints is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                np.save(os.path.join(DATA_PATH, letter, f"{count}.npy"), keypoints)
                count += 1

            cv2.putText(frame, f'Recording "{letter}": {count}/{SAMPLES_PER_LETTER}', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or window_closed(WINDOW_NAME):
                cap.release()
                cv2.destroyAllWindows()
                print("Closed by user.")
                exit()

cap.release()
cv2.destroyAllWindows()
print("\n[SUCCESS] All letters collected!")