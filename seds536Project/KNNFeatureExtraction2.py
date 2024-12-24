import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

# Mediapipe settings
mp_hands = mp.solutions.hands

# Paths
input_dir = "detected_hands"  # Directory containing detected hand images

# Data storage
landmarks = []
labels = []

# Helper Functions
def preprocess_image(image_path):
    """
    Preprocess the image for Mediapipe:
    - Read image
    - Convert to RGB format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks to make them independent of position and scale.
    """
    wrist = hand_landmarks[0]  # Wrist is the reference point
    # Calculate the distance of each landmark from the wrist
    distances = [np.linalg.norm([point.x - wrist.x, point.y - wrist.y]) for point in hand_landmarks]
    scale = max(distances)  # Max distance for normalization

    normalized_landmarks = []
    for point in hand_landmarks:
        normalized_landmarks.append([
            (point.x - wrist.x) / scale,  # X normalized
            (point.y - wrist.y) / scale,  # Y normalized
            (point.z - wrist.z) / scale   # Z normalized
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
    angle = np.arctan2(vector[1], vector[0])  # Calculate angle in radians
    return angle

def compute_thumb_pinky_vector(normalized_landmarks):
    """
    Compute the vector from Thumb Tip (4) to Pinky Tip (20).
    """
    return normalized_landmarks[20] - normalized_landmarks[4]

# Mediapipe hand detection
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    # Calculate total number of images to process
    total_images = sum([len(files) for r, d, files in os.walk(input_dir)])
    processed_count = 0

    # Iterate over each label directory (e.g., "1" and "2")
    for label in ["1", "2"]:  # Assuming two classes: 1 and 2
        label_dir = os.path.join(input_dir, label)

        for file_name in os.listdir(label_dir):
            # Filter only image files (e.g., .jpg, .jpeg, .png)
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            file_path = os.path.join(label_dir, file_name)
            try:
                # Preprocess the image
                image, image_rgb = preprocess_image(file_path)

                # Process image with Mediapipe
                result = hands.process(image_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Normalize landmarks
                        normalized = normalize_landmarks(hand_landmarks.landmark)

                        # Extract additional features
                        thumb_pinky_distance = compute_thumb_pinky_distance(normalized)
                        thumb_wrist_angle = compute_thumb_wrist_angle(normalized)
                        thumb_pinky_vector = compute_thumb_pinky_vector(normalized)

                        # Combine selected features into a single feature list
                        features = (
                            normalized.flatten().tolist() +  # Flatten all landmark coordinates
                            [thumb_pinky_distance, thumb_wrist_angle] +  # Add distance and angle
                            thumb_pinky_vector.tolist()  # Add vector components
                        )

                        # Append the features and corresponding label to the data storage
                        landmarks.append(features)
                        labels.append(0 if label == "1" else 1)  # Label 1 = 0, Label 2 = 1
                else:
                    print(f"No landmarks found in image: {file_path}")
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")

            # Debug progress
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_images:
                print(f"Processed {processed_count}/{total_images} images.")

# Convert to numpy arrays
X = np.array(landmarks)  # Shape: (Num_samples, 68)
y = np.array(labels)     # Shape: (Num_samples,)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Save to .npz files
np.savez("landmark_train_final.npz", data=X_train, labels=y_train)
np.savez("landmark_val_final.npz", data=X_val, labels=y_val)
np.savez("landmark_test_final.npz", data=X_test, labels=y_test)

print("Final normalized landmark data saved successfully!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
