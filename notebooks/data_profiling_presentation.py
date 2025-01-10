import math
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import io
import zipfile
import tarfile
from collections import Counter
import sys
import json
from matplotlib.colors import LogNorm


def open_and_measure_lmdbs(data_paths: list, max_size):
    n_imgs_per_bin: np.ndarray = np.full((max_size, max_size), fill_value= 0, dtype=int)
    n_images_too_large: int = 0
    minmax_dict_global: dict[str, dict] = {} # dict to store minmax values for each path
    # run the following code for each path in data_path
    for data_path in data_paths:
        minmax_dict_tmp: dict[str, int] = {"min_width": sys.maxsize, "max_width": -1, "min_height": sys.maxsize, "max_height": -1, "n_images": 0}
       
        n_images_too_large, minmax_dict_tmp = process_lmdb(data_path, n_imgs_per_bin, max_size, minmax_dict_tmp)
        
        print(f"Profiling done for: {data_path}")
        print(minmax_dict_tmp)
        minmax_dict_global[data_path] = minmax_dict_tmp
    
    heights = n_imgs_per_bin.sum(axis=0)
    widths = n_imgs_per_bin.sum(axis=1)

    height_stats = {
            'median': np.median(heights),
            'mean': np.mean(heights),
            'std': np.std(heights),
        }

    width_stats = {
            'median': np.median(widths),
            'mean': np.mean(widths),
            'std': np.std(widths),
        }
    
    print(f"Height stats: \n{height_stats}\n")
    print(f"Width stats: \n{width_stats}\n")
    print(f"Number of images too large: {n_images_too_large}")
    
    return n_imgs_per_bin

def process_lmdb(data_path, n_imgs_per_bin, max_size:int, minmax_dict:dict):
    path = os.path.join(data_path)
    env_imgs: lmdb.Environment = lmdb.open(path, readonly=True)

    n_images_too_large = 0
    
    with env_imgs.begin() as txn_imgs:
        cursor_imgs = txn_imgs.cursor()
        
        for (img_key, img_value) in cursor_imgs:

            img = Image.open(io.BytesIO(img_value))

            # print('Image format = ', img.size)

            width, height = img.size
            minmax_dict["min_width"] = min(minmax_dict["min_width"], width)
            minmax_dict["max_width"] = max(minmax_dict["max_width"], width)
            minmax_dict["min_height"] = min(minmax_dict["min_height"], height)
            minmax_dict["max_height"] = max(minmax_dict["max_height"], height)
            minmax_dict["n_images"] += 1

            if width < max_size and height < max_size:
               n_imgs_per_bin[ width, height] += 1
            else:
                n_images_too_large += 1
                
    return n_images_too_large, minmax_dict

def create_heatmap(widths, heights, num_images=-1, bins=100, folder_name=""):
    plt.hist2d(widths, heights, bins)
    plt.colorbar()
    plt.xlabel('Width (in px)')
    plt.ylabel('Height (in px)')
    plt.title(f'{folder_name} Size Distribution (Log Scale) ({num_images} images)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def create_heatmap_array(n_imgs_per_bin, path=os.path.join(os.getcwd(), "output", "heatmap.png")):
    plt.imshow(n_imgs_per_bin,  cmap='viridis', origin='lower', norm=LogNorm())
    plt.colorbar()
    plt.xlabel('Width (in px)')
    plt.ylabel('Height (in px)')
    plt.title('Size Distribution')
    # plt.yscale('log')
    # plt.xscale('log')

    plt.savefig(path)
    plt.clf()

if __name__ == "__main__":

    lmdb_paths_labelled:list[str] = [
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/datasciencebowl/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/FlowCamNet/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ISIISNet/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/seanoe_uvp_labeled/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/UVPEC/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooCamNet/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooScanNet/images",
    ]
    lmdb_paths_unlabelled:list[str] = [
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/datasciencebowl/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/pisco/images",
        # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/IFCB_downloader/images",
        "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/seanoe_uvp_unlabeled/images",
    ]

    test_path = ["/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/datasciencebowl/images"]
    print(os.getcwd())

    array_unlabelled = open_and_measure_lmdbs(lmdb_paths_labelled, max_size=1000)
    path_heatmap_unlabelled = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_unlabelled.png"
    create_heatmap_array(array_unlabelled, path_heatmap_unlabelled)

    path_unlabelled_zoomed = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_unlabelled_zoomed.png"
    create_heatmap_array(array_unlabelled[:300, :300], path_unlabelled_zoomed)

    array_labelled = open_and_measure_lmdbs(lmdb_paths_unlabelled, max_size=1000)
    path_heatmap_labelled = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_labelled.png"
    create_heatmap_array(array_labelled, path_heatmap_labelled)

    path_labelled_zoomed = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_labelled_zoomed.png"
    create_heatmap_array(array_labelled[:300, :300], path_labelled_zoomed)


