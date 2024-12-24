import os
import cv2
import mediapipe as mp

# Mediapipe settings
mp_hands = mp.solutions.hands

# Paths
input_dir = "detected_hands"  # Directory containing images processed by Mediapipe
output_dir = "yolo_data"  # Directory to save data in YOLO format
os.makedirs(output_dir, exist_ok=True)

def create_bounding_box(hand_landmarks, image_width, image_height):
    """
    Create a bounding box from hand landmarks.
    The box coordinates are normalized relative to the image dimensions.
    Args:
        hand_landmarks: List of hand landmark points from Mediapipe.
        image_width: Width of the image.
        image_height: Height of the image.
    Returns:
        Normalized bounding box (x_center, y_center, width, height).
    """
    # Extract x and y coordinates of all landmarks
    x_coords = [landmark.x * image_width for landmark in hand_landmarks]
    y_coords = [landmark.y * image_height for landmark in hand_landmarks]

    # Find min and max coordinates to create a bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate the center, width, and height of the bounding box (normalize values)
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return x_center, y_center, width, height

# Mediapipe hand detection and YOLO format data creation
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    for label in ["1", "2"]:  # Assuming two classes: 1 (label 0) and 2 (label 1)
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)  # Create output directories

        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            try:
                # Load the image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error loading image: {file_path}")
                    continue

                # Get image dimensions
                image_height, image_width, _ = image.shape

                # Convert the image to RGB format for Mediapipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image_rgb)

                # Process hand landmarks if detected
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Generate bounding box in YOLO format
                        x_center, y_center, width, height = create_bounding_box(
                            hand_landmarks.landmark, image_width, image_height
                        )

                        # Determine class ID based on the label (1 -> 0, 2 -> 1)
                        yolo_label = 0 if label == "1" else 1

                        # Save YOLO format data to a text file
                        output_txt_path = os.path.join(output_label_dir, f"{os.path.splitext(file_name)[0]}.txt")
                        with open(output_txt_path, "w") as f:
                            f.write(f"{yolo_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                else:
                    # If no hands are detected in the image, print a warning
                    print(f"No hands detected in image: {file_path}")
            except Exception as e:
                # Handle any errors during processing
                print(f"Error processing image {file_path}: {e}")

print("Bounding boxes created and YOLO format data saved.")
