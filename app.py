import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
# Let's lower the minimum detection confidence to be less strict
hands = mp_hands.Hands(min_detection_confidence=0.7) 
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror view
    frame = cv.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hand landmarks
    results = hands.process(rgb_frame)

    # --- DEBUGGING STEP ---
    # Check if any hands were detected
    if results.multi_hand_landmarks:
        print("Hand Detected!") # This should print when a hand is seen
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections on the original frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    # --- END DEBUGGING STEP ---

    # Display the frame
    cv2.imshow("Hand Landmark Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()