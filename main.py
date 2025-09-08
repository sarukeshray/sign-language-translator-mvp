import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_model.h5')

# Actions that the model can recognize
actions = np.array(['hello', 'thank_you', 'yes', 'no'])

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables for prediction logic
sequence = []
sentence = []
threshold = 0.8 # Confidence threshold

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Flip frame for mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract keypoints
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            sequence.append(keypoints)
    else:
        # If no hand is detected, you might want to reset the sequence
        # For now, we'll just let it be, but this is an area for improvement
        pass
    
    # Keep the sequence to the last 30 frames
    sequence = sequence[-30:]
    
    # Check if we have enough frames for a prediction
    if len(sequence) == 30:
        # Predict the action
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        prediction = actions[np.argmax(res)]
        
        # Display the prediction if confidence is high enough
        if res[np.argmax(res)] > threshold:
            sentence.append(prediction)

            # Keep the sentence to the last 5 words for display
            if len(sentence) > 5:
                sentence = sentence[-5:]

    # --- Visualization Logic ---
    # Draw a status box at the bottom
    cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
    # Display the sentence
    cv2.putText(frame, ' '.join(sentence), (3, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the window
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()