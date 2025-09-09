from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# --- ML MODEL SETUP (remains the same) ---
model = load_model('sign_language_model.h5')
actions = np.array(['hello', 'thank_you', 'yes', 'no'])
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Global variables for prediction logic ---
sequence = []
threshold = 0.8

# --- FLASK ROUTE ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SOCKETIO EVENT HANDLER (Simplified) ---
@socketio.on('image')
def handle_image(data_image):
    """Receives an image, predicts the sign in English, and sends it back."""
    global sequence
    
    # Decode the image
    sbuf = io.BytesIO(base64.b64decode(data_image.split(',')[1]))
    pil_image = Image.open(sbuf)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Process frame and get prediction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    keypoints = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        
    sequence.append(keypoints)
    sequence = sequence[-30:]

    prediction = ""
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            prediction = actions[np.argmax(res)]
    
    # Send the single English prediction back to the client
    emit('response', {'word': prediction})

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    socketio.run(app, debug=True)