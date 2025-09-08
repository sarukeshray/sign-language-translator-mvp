import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Path to save the data
DATA_PATH = "data"

# Signs to collect
signs = ['hello', 'thank_you', 'yes', 'no']
# Number of sequences (videos) for each sign
num_sequences = 30
# Number of frames per sequence
sequence_length = 30

# Create folders for each sign if they don't exist
for sign in signs:
    # Use a loop to create directories for each sequence
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
        except FileExistsError:
            pass

print("Starting data collection...")

# Main Loop
for sign in signs:
    for sequence in range(num_sequences):
        for frame_num in range(sequence_length):
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Displaying information on the screen
            if frame_num == 0:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting frames for {sign} - Seq {sequence}', (15, 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(2000) # Wait for 2 seconds
            else:
                cv2.putText(frame, f'Collecting frames for {sign} - Seq {sequence}', (15, 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)

            # NEW LOGIC: Save zeros if no hand is detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            else:
                # If no hand is detected, save an array of 63 zeros (21 landmarks * 3 coords)
                keypoints = np.zeros(21 * 3)

            # Save the keypoints
            npy_path = os.path.join(DATA_PATH, sign, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()