import lmdb
import cv2
import numpy as np
import os
import glob
from PIL import Image
import io

# Define the parent directory where your LMDB files are located
PARENT_DIR = '/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/' 

def decode_image(image_bytes):
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print("Failed to decode image:", e)
        return None

def read_images_from_lmdb(lmdb_path, num_images=100):
    print(f"\nReading LMDB: {lmdb_path}")
    images = []
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_images:
                break
            img = decode_image(value)
            if img:
                images.append(np.array(img))
    env.close()
    return images

def compute_white_pixel_percentage(images):
    """
    Computes the average percentage of white pixels (>240 intensity) per image.
    """
    if not images:
        return 0.0

    white_pixel_percentages = []

    for image_np in images:
        # Convert to grayscale if needed
        if len(image_np.shape) > 2 and image_np.shape[2] > 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = np.squeeze(image_np)

        total_pixels = image_np.size
        white_pixels = np.sum(image_np > 250)
        percentage = (white_pixels / total_pixels) * 100
        white_pixel_percentages.append(percentage)

    print("num_images",len(white_pixel_percentages))
    if len(white_pixel_percentages) == 0:
        return None
    return np.mean(white_pixel_percentages)

def main():
    lmdb_paths = sorted(glob.glob(os.path.join(PARENT_DIR, "**"), recursive=True))
    lmdb_img_files = [p for p in lmdb_paths if p.endswith(("imgs", "images")) and os.path.isdir(p)]

    print(f"Found {len(lmdb_img_files)} LMDB image directories.")

    for lmdb_path in lmdb_img_files:
        rel_path = os.path.relpath(lmdb_path, PARENT_DIR)
        lmdb_name = "_".join(rel_path.split(os.sep))

        avg_white_percent = compute_white_pixel_percentage_streaming(lmdb_path, num_images=1000000)
        print(f"{lmdb_name}: Average white pixel percentage = {avg_white_percent:.2f}%")

def compute_white_pixel_percentage_streaming(lmdb_path, num_images=100):
    print(f"\nReading LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    total_percentage = 0.0
    valid_images = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_images:
                break

            img = decode_image(value)
            if img is None:
                continue

            image_np = np.array(img)

            # Convert to grayscale if needed
            if len(image_np.shape) > 2 and image_np.shape[2] > 1:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 1:
                image_np = np.squeeze(image_np)

            total_pixels = image_np.size
            white_pixels = np.sum(image_np > 240)
            percentage = (white_pixels / total_pixels) * 100

            total_percentage += percentage
            valid_images += 1

    env.close()
    print("num_images",valid_images)
    if valid_images == 0:
        return 0.0
    
    return total_percentage / valid_images

if __name__ == "__main__":
    main()
