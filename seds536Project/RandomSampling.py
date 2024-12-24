import os
import shutil
import random

# Paths
input_dir = "gesture_images"  # Original dataset directory
output_dir = "gesture_images_small"  # Reduced dataset directory

# Number of images to sample per class
sample_size = 7000

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Debug message: Start of the process
print("Starting dataset reduction process...")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Target sample size per class: {sample_size}")

# Process each class
for label in ["1", "2"]:  # Assuming you have "1" and "2" as class folders
    label_dir = os.path.join(input_dir, label)
    output_label_dir = os.path.join(output_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get all images in the class directory
    images = os.listdir(label_dir)
    total_images = len(images)
    print(f"Class {label}: Found {total_images} images.")

    # Check if the class already has fewer images than the sample size
    if total_images <= sample_size:
        print(f"Class {label} already has fewer than or equal to {sample_size} images. Skipping sampling.")
        for image in images:
            src_path = os.path.join(label_dir, image)
            dest_path = os.path.join(output_label_dir, image)
            shutil.move(src_path, dest_path)
        continue

    # Randomly sample the specified number of images
    print(f"Sampling {sample_size} images from class {label}...")
    sampled_images = random.sample(images, sample_size)

    # Move sampled images to the new directory
    for i, image in enumerate(sampled_images, start=1):
        src_path = os.path.join(label_dir, image)
        dest_path = os.path.join(output_label_dir, image)
        shutil.move(src_path, dest_path)
        
        # Debug progress every 500 images
        if i % 500 == 0:
            print(f"Class {label}: {i}/{sample_size} images moved...")

    print(f"Class {label}: Sampling complete. {sample_size} images moved to {output_label_dir}.")

# Debug message: End of the process
print("Dataset reduction complete! All classes processed.")
