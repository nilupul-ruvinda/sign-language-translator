import cv2
import string
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

LETTERS = list(string.ascii_uppercase)
WINDOW_NAME = 'Alphabet Sign Translator'
model = load_model('models/letter_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_hand_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        # 1. Center relative to wrist (landmark 0)
        coords = coords - coords[0]
        
        # 2. Scale normalize
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist
            
        # 3. Fingertip-to-thumb distances
        tips = [4, 8, 12, 16, 20]
        thumb_tip = coords[4]
        tip_distances = [np.linalg.norm(coords[t] - thumb_tip) for t in tips[1:]]
        
        return np.concatenate([coords.flatten(), tip_distances])  # 67 features
    return None

def window_closed(win_name):
    return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1

word = ""
current_letter = ""
hold_frames = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        keypoints = extract_hand_landmarks(results)
        detected_display = ""

        if keypoints is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            prediction = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            idx = np.argmax(prediction)
            confidence = prediction[idx]

            if confidence > 0.75:
                detected_char = LETTERS[idx]
                detected_display = f"{detected_char} ({confidence*100:.0f}%)"
                if detected_char == current_letter:
                    hold_frames += 1
                    if hold_frames == 20:
                        word += detected_char
                else:
                    current_letter = detected_char
                    hold_frames = 0
        else:
            current_letter = ""
            hold_frames = 0

        cv2.rectangle(frame, (0, 0), (640, 50), (30, 30, 30), -1)
        cv2.putText(frame, f'Sign: {detected_display} | Word: {word}', (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):                     # Press C to clear word
            word = ""
        elif key == 8 and len(word) > 0:        # Press Backspace to delete last character
            word = word[:-1]
        elif key == 32:                         # Press Spacebar to add space
            word += " "
        elif key == ord('q') or window_closed(WINDOW_NAME):
            break

cap.release()
cv2.destroyAllWindows()