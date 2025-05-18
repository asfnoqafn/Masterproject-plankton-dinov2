import lmdb
import cv2
import numpy as np
import os
import glob
from PIL import Image
import io

# Define the parent directory where your LMDB files are located
# Make sure to change this to your actual parent directory
PARENT_DIR = '/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/' 
OUTPUT_PARENT_DIR = './output_pngs3' # Directory to save the output PNGs

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
                # Store the decoded image and its key (as a string)
                images.append((key.decode('utf-8', errors='ignore'), img))
    env.close()
    return images

def save_images_as_png(images, lmdb_name, output_parent_dir):
    """
    Saves images as PNG files using their original LMDB keys as filenames.

    Args:
        images (list of tuples): List of (key, PIL image) pairs.
        lmdb_name (str): Name of LMDB directory.
        output_parent_dir (str): Output directory base path.
    """
    if not images:
        print(f"No images to save for {lmdb_name}")
        return

    output_dir = os.path.join(output_parent_dir, lmdb_name + "_pngs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(images)} images to {output_dir}")

    for key, image in images:
        # Convert to NumPy array
        image_np = np.array(image)


        # Convert to grayscale if needed
        if len(image_np.shape) > 2 and image_np.shape[2] > 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = np.squeeze(image_np)

        # Clean key to make it filesystem-safe
        safe_key = "".join(c if c.isalnum() else "_" for c in key)

        img_filename = os.path.join(output_dir, f"{lmdb_name}_{safe_key}.png")
        try:
            cv2.imwrite(img_filename, image_np)
        except Exception as e:
            print(f"Error saving image {img_filename}: {e}")

def main():
    """
    Finds LMDB image directories, reads the first 100 images, and saves them as PNGs.
    """
    # Find all *.imgs or *.images LMDB files recursively
    lmdb_paths = sorted(
        glob.glob(os.path.join(PARENT_DIR, "**"), recursive=True)
    )
    lmdb_img_files = [p for p in lmdb_paths if p.endswith(("imgs", "images")) and os.path.isdir(p)]

    print(f"Found {len(lmdb_img_files)} LMDB image directories.")

    # Create the main output directory if it doesn't exist
    os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

    for lmdb_path in lmdb_img_files:
        print(f"Processing LMDB: {lmdb_path}")
        images = read_images_from_lmdb(lmdb_path, num_images=100)
        if images:
            rel_path = os.path.relpath(lmdb_path, PARENT_DIR)
            lmdb_name = "_".join(rel_path.split(os.sep))
            save_images_as_png(images, lmdb_name, OUTPUT_PARENT_DIR)
        else:
            print(f"No valid images found or read from {lmdb_path}")

if __name__ == "__main__":
    main()