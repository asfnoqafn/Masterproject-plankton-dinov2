import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import io
import zipfile
import tarfile
from collections import Counter
import json

def append_height_width_threshold(height, width, heights, widths):
    threshold = 400
    if width < threshold and height < threshold:
        heights.append(height)
        widths.append(width)

def append_height_width(height, width, heights, widths):
    heights.append(height)
    widths.append(width)

def profile_by_label(data_path, archive_type):
    assert archive_type == 'lmdb', "archive_type needs to be lmdb"
    env_labels = lmdb.open(data_path, readonly=True)
    label_dict = {}
    with open("/Users/Johann/Masterproject-plankton-dinov2/data/plankton/-TEST_label_map.json", "r") as f:
        class_map = json.load(f)
    with env_labels.begin() as txn_labels:

        cursor_labels = txn_labels.cursor()
        for (label_key, label) in cursor_labels:
            label_str = int.from_bytes(label, byteorder="little") 
            # print(label_key, label)
            
            # label_dict[label_str] = label_dict.get(label_str, 0) + 1
            if label_str not in label_dict.keys():
                label_dict[label_str] = 1
            else:
                label_dict[label_str] += 1

    label_dict_str = {}

    for key in label_dict.keys():
        label_dict_str[class_map[str(key)]] = label_dict[key]

        # Output the resulting dictionary
    print(label_dict_str)


def open_and_measure_data(data_path, archive_type, threshold=-1):
    lmdb_labels_path = "/home/hk-project-p0021769/hgf_col5747/data/plankton/-TRAIN_labels"
    heights = []
    widths = []
    num_images = 0
    folder_name = ''
    if archive_type == 'lmdb':

        folder_name = data_path.split('/')[-1]
        
        env_imgs = lmdb.open(data_path, readonly=True)

        with env_imgs.begin() as txn_imgs:
            cursor_imgs = txn_imgs.cursor()
            
            # label_dict = Counter(cursor_labels)
            # print('label_dict:', label_dict)
            
            for (img_key, img_value) in cursor_imgs:
                img = Image.open(io.BytesIO(img_value))

                # print('Image format = ', img.size)

                width, height = img.size

                num_images += 1
                if width < 400 and height < 400:
                    widths.append(width)
                    heights.append(height)


    elif archive_type == 'zip':
        folder_name = os.path.basename(data_path)
        with zipfile.ZipFile(data_path, 'r') as zip_file:
            image_files = [name for name in zip_file.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                with zip_file.open(img_file) as img_data:
                    img = Image.open(img_data)
                    width, height = img.size
                    num_images += 1
                    if width < 400 and height < 400:
                        widths.append(width)
                        heights.append(height)


    elif archive_type == 'tar':
        folder_name = os.path.basename(data_path)
        with tarfile.open(data_path, 'r') as tar_file:
            image_files = [member for member in tar_file.getmembers() if member.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                img_data = tar_file.extractfile(img_file)
                if img_data:
                    img = Image.open(img_data)
                    width, height = img.size
                    num_images += 1
                    if width < 400 and height < 400:
                        widths.append(width)
                        heights.append(height)

    elif archive_type == 'file':
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    img = Image.open(file_path)
                    width, height = img.size
                    num_images += 1
                    if threshold > 0 and width < threshold and height < threshold:
                        widths.append(width)
                        heights.append(height)
                    else:
                        widths.append(width)
                        heights.append(height)
    else:
        raise ValueError("Unsupported archive type. Use 'zip' or 'tar'.")

    height_stats = {
            'median': np.median(heights),
            'mean': np.mean(heights),
            'std': np.std(heights),
            'min': np.amin(heights),
            'max': np.amax(heights)
        }

    width_stats = {
            'median': np.median(widths),
            'mean': np.mean(widths),
            'std': np.std(widths),
            'min': np.min(widths),
            'max': np.max(widths)
        }
    
    return height_stats, width_stats, heights, widths

def create_heatmap(widths, heights, num_images=-1, bins=100, folder_name=""):
    plt.hist2d(widths, heights, bins)
    plt.colorbar()
    plt.xlabel('Width (in px)')
    plt.ylabel('Height (in px)')
    plt.title(f'{folder_name} Size Distribution (Log Scale) ({num_images} images)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    profile_by_label(
        "/Users/Johann/Masterproject-plankton-dinov2/data/plankton/-VAL_labels",
        "lmdb",
    )
