import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
input_dir = "detected_hands"  # Directory containing images and labels
output_base_dir = "yolo_split_data"  # Directory to save split data
os.makedirs(output_base_dir, exist_ok=True)

# Train/val/test ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Validate that ratios sum to 1
assert round(train_ratio + val_ratio + test_ratio, 5) == 1, "Ratios must sum to 1!"

# Subdirectories for splits
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, "labels"), exist_ok=True)

print("Initialized output directories for train, validation, and test splits.")

# Iterate through each class folder (e.g., "1", "2")
for label in ["1", "2"]:
    print(f"Processing class: {label}")

    class_dir = os.path.join(input_dir, label)
    images = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
    annotations = [f.replace(".jpg", ".txt") for f in images]  # Corresponding .txt files

    # Combine images and labels as pairs
    data_pairs = list(zip(images, annotations))

    if len(data_pairs) == 0:
        print(f"Warning: No data found for class {label}. Skipping.")
        continue

    print(f"Found {len(data_pairs)} pairs for class {label}.")

    # Split data into train, val, and test
    train_data, temp_data = train_test_split(data_pairs, test_size=(val_ratio + test_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    print(f"Class {label}: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

    # Function to copy data to respective directories
    def copy_data(data, split):
        for image_file, label_file in data:
            src_image = os.path.join(class_dir, image_file)
            src_label = os.path.join(class_dir, label_file)
            dst_image = os.path.join(output_base_dir, split, "images", image_file)
            dst_label = os.path.join(output_base_dir, split, "labels", label_file)

            try:
                shutil.copy(src_image, dst_image)
                shutil.copy(src_label, dst_label)
            except Exception as e:
                print(f"Error copying {image_file} or {label_file}: {e}")

    # Copy to respective directories
    copy_data(train_data, "train")
    copy_data(val_data, "val")
    copy_data(test_data, "test")

print("Dataset successfully split into train, validation, and test sets.")
