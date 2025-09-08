import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- 1. Load and Process Data ---

# Path for exported data, numpy arrays
DATA_PATH = "data" 

# Actions that we try to detect
actions = np.array(['hello', 'thank_you', 'yes', 'no'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# A mapping from actions (strings) to numbers
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the saved numpy array for each frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
# One-hot encode the labels
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. Build the LSTM Model ---

# A sequential model is a linear stack of layers
model = Sequential()
# Add LSTM layers. return_sequences=True is needed for stacking LSTM layers
# The input shape is (30 frames, 63 landmarks per frame)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
# Add Dense layers for further processing
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# The final layer uses 'softmax' activation to output a probability distribution over the actions
model.add(Dense(actions.shape[0], activation='softmax'))

# --- 3. Compile and Train the Model ---

# Compile the model with the Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting model training...")
# Train the model. An epoch is one complete pass through the entire training dataset.
model.fit(X_train, y_train, epochs=30)

model.summary()

# --- 4. Save the Model ---
model.save('sign_language_model.h5')
print("Model trained and saved as sign_language_model.h5")