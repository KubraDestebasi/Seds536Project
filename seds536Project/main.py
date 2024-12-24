import cv2
import mediapipe as mp
from ultralytics import YOLO
import pickle
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np
import pygame
import math

# Initialize Mediapipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load KNN Model and Scaler
with open("KNNversion2.pkl", "rb") as f:
    knn_model = pickle.load(f)

try:
    with open("scaler_version2.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    scaler = None
    print("No scaler found. Ensure data normalization consistency.")

# Load YOLO Model
yolo_model = YOLO("datasets/GestureProject/gesture_exp/weights/best.pt")

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Initialize volume level (0.0 to 1.0)
current_volume = volume.GetMasterVolumeLevelScalar()

# Confidence threshold for KNN predictions
confidence_threshold = 0.75  # Adjust this as needed

# Start the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for background music
pygame.mixer.init()
pygame.mixer.music.load("background_music.mp3")  # Replace with your music file
pygame.mixer.music.set_volume(0.2)  # Set background music volume
pygame.mixer.music.play(-1)  # Loop the music indefinitely

# Variables to control the prediction mode and display the active model
use_knn = False  # Default to YOLO
gesture_label = None

# Smooth volume transition
def set_smooth_volume(target_volume, step=0.05):
    global current_volume
    delta = target_volume - current_volume
    if abs(delta) > step:
        current_volume += step if delta > 0 else -step
        volume.SetMasterVolumeLevelScalar(current_volume, None)
    else:
        current_volume = target_volume
        volume.SetMasterVolumeLevelScalar(current_volume, None)

# Helper Functions for Feature Extraction
def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks to make them independent of position and scale.
    """
    wrist = hand_landmarks[0]  # Wrist is the reference point
    distances = [np.linalg.norm([point[0] - wrist[0], point[1] - wrist[1]]) for point in hand_landmarks]
    scale = max(distances)  # Max distance for normalization

    normalized_landmarks = []
    for point in hand_landmarks:
        normalized_landmarks.append([
            (point[0] - wrist[0]) / scale,  # X normalized
            (point[1] - wrist[1]) / scale,  # Y normalized
            (point[2] - wrist[2]) / scale   # Z normalized
        ])
    return np.array(normalized_landmarks)

def compute_thumb_pinky_distance(normalized_landmarks):
    """
    Compute the distance between the Thumb Tip (4) and Pinky Tip (20).
    """
    return np.linalg.norm(normalized_landmarks[4] - normalized_landmarks[20])

def compute_thumb_wrist_angle(normalized_landmarks):
    """
    Compute the angle between the Thumb Tip (4) and Wrist (0).
    """
    vector = normalized_landmarks[4] - normalized_landmarks[0]
    angle = math.atan2(vector[1], vector[0])  # Calculate angle in radians
    return angle

def compute_thumb_pinky_vector(normalized_landmarks):
    """
    Compute the vector from Thumb Tip (4) to Pinky Tip (20).
    """
    return normalized_landmarks[20] - normalized_landmarks[4]

# Main loop for webcam and predictions
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirrored view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand landmark detection using Mediapipe
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract and normalize all hand landmarks
            normalized_landmarks = normalize_landmarks([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Compute additional features
            thumb_pinky_distance = compute_thumb_pinky_distance(normalized_landmarks)
            thumb_wrist_angle = compute_thumb_wrist_angle(normalized_landmarks)
            thumb_pinky_vector = compute_thumb_pinky_vector(normalized_landmarks)
            
            # Combine all features into a single feature vector
            features = normalized_landmarks.flatten().tolist() + [
                thumb_pinky_distance,  # Distance feature
                thumb_wrist_angle,     # Angle feature
                *thumb_pinky_vector.tolist()  # Vector components (X, Y, Z)
            ]

            # Convert to numpy array and reshape for the scaler and model
            features = np.array(features).reshape(1, -1)

            # Scale the features if a scaler is available
            if scaler:
                features = scaler.transform(features)

            # Perform prediction based on the active model
            if use_knn:
                # Predict using KNN
                prediction = knn_model.predict(features)
                confidence = knn_model.predict_proba(features)[0].max()
                gesture_label = prediction[0] if confidence > confidence_threshold else None
            else:
                # Predict using YOLO
                yolo_results = yolo_model.predict(source=rgb_frame, conf=0.25, verbose=False)
                if len(yolo_results[0].boxes):
                    gesture_label = int(yolo_results[0].boxes[0].cls)
                else:
                    gesture_label = None  # No gesture detected

            # Perform actions based on the gesture label
            if gesture_label == 0:
                set_smooth_volume(1)  # Smoothly increase volume to 1
                cv2.putText(frame, "Thumbs up: Volume Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture_label == 1:
                set_smooth_volume(0)  # Smoothly decrease volume to 0
                cv2.putText(frame, "Thumbs down: Volume Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Unknown gesture when hand is detected but not recognized
                cv2.putText(frame, "Unknown Gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        # No hands detected
        cv2.putText(frame, "No Hands Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the active model on the screen
    model_name = "KNN" if use_knn else "YOLO"
    cv2.putText(frame, f"Active Model: {model_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video feed with annotations
    cv2.imshow("Gesture Recognition", frame)

    # Handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):  # Switch to KNN model
        use_knn = True
    elif key == ord("n"):  # Switch to YOLO model
        use_knn = False
    elif key == ord("q"):  # Quit the application
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
