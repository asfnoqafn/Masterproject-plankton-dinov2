import argparse
import json
import os
import sys

import imageio as io
import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

BASE_DIR = "/fast/AG_Kainmueller/data/pan_m"  # max cluster path
MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB


def mibi_breast_naming_conv(fov_path):
    base_dir_ = os.path.join(BASE_DIR, "mibi_breast")
    fov_name = os.path.basename(fov_path)
    deepcell_output_dir = os.path.join(base_dir_, "segmentation_data")
    return os.path.join(
        deepcell_output_dir,
        "deepcell_output",
        fov_name + "_feature_0.tif",
    )


def mibi_decidua_naming_conv(fov_path):
    base_dir_ = os.path.join(BASE_DIR, "mibi_decidua")
    fov_name = os.path.basename(fov_path)
    deepcell_output_dir = os.path.join(base_dir_, "segmentation_data")
    return os.path.join(
        deepcell_output_dir,
        fov_name + "_segmentation_labels.tiff",
    )


def vectra_colon_naming_conv(fname):
    return os.path.join(
        BASE_DIR,
        "vectra_colon",
        "segmentation",
        fname + "feature_0.ome.tif",
    )


def vectra_pancreas_naming_conv(fname):
    return os.path.join(
        BASE_DIR,
        "vectra_pancreas",
        "segmentation",
        fname + "feature_0.ome.tif",
    )


def codex_colon_naming_conv(fname):
    fov, reg = fname.split("_")[:2]
    fov_path = os.path.join(BASE_DIR, "codex_colon", "masks", fov)
    images = os.listdir(fov_path)
    labels = [img for img in images if "_labeled" in img]
    labels = [img for img in labels if reg in img]
    label_fname = labels[0]
    return os.path.join(os.path.normpath(fov_path), label_fname)


naming_convention_dict = {
    "mibi_breast": mibi_breast_naming_conv,
    "mibi_decidua": mibi_decidua_naming_conv,
    "vectra_colon": vectra_colon_naming_conv,
    "vectra_pancreas": vectra_pancreas_naming_conv,
    "codex_colon": codex_colon_naming_conv,
}

selected_channels = {
    "mibi_breast": [
        "Calprotectin.tiff",
        "CD11c.tiff",
        "CD14.tiff",
        "CD163.tiff",
        "CD20.tiff",
        "CD3.tiff",
        "CD31.tiff",
        "CD38.tiff",
        "CD4.tiff",
        "CD45.tiff",
        "CD45RB.tiff",
        "CD45RO.tiff",
        "CD56.tiff",
        "CD57.tiff",
        "CD68.tiff",
        "CD69.tiff",
        "CD8.tiff",
        "ChyTr.tiff",
        "CK17.tiff",
        "Collagen1.tiff",
        "ECAD.tiff",
        "FAP.tiff",
        "Fibronectin.tiff",
        "FOXP3.tiff",
        "GLUT1.tiff",
        "H3K27me3.tiff",
        "H3K9ac.tiff",
        "HLA1.tiff",
        "HLADR.tiff",
        "IDO.tiff",
        "Ki67.tiff",
        "LAG3.tiff",
        "PD1.tiff",
        "PDL1.tiff",
        "SMA.tiff",
        "TBET.tiff",
        "TCF1.tiff",
        "TIM3.tiff",
        "Vim.tiff",
    ],
    "mibi_decidua": [
        "CD11b.tif",
        "CD11c.tif",
        "CD14.tif",
        "CD16.tif",
        "CD163.tif",
        "CD20.tif",
        "CD206.tif",
        "CD3.tif",
        "CD31.tif",
        "CD4.tif",
        "CD44.tif",
        "CD45.tif",
        "CD56.tif",
        "CD57.tif",
        "CD68.tif",
        "CD8.tif",
        "CD80.tif",
        "CK7.tif",
        "DCSIGN.tif",
        "Ecad.tif",
        "FoxP3.tif",
        "Galectin9.tif",
        "GrB.tif",
        "H3.tif",
        "HLADR.tif",
        "HLAG.tif",
        "HO1.tif",
        "ICOS.tif",
        "IDO.tif",
        "iNOS.tif",
        "Ki67.tif",
        "PD1.tif",
        "PDL1.tif",
        "SMA.tif",
        "TIGIT.tif",
        "TIM3.tif",
        "Tryptase.tif",
        "VIM.tif",
    ],
    "vectra_colon": [
        "CD3.ome.tif",
        "CD8.ome.tif",
        "DAPI.ome.tif",
        "Foxp3.ome.tif",
        "ICOS.ome.tif",
        "panCK+CK7+CAM5.2.ome.tif",
        "PD-L1.ome.tif",
    ],
    "vectra_pancreas": [
        "CD40-L.ome.tif",
        "CD40.ome.tif",
        "CD8.ome.tif",
        "DAPI.ome.tif",
        "panCK.ome.tif",
        "PD-1.ome.tif",
        "PD-L1.ome.tif",
    ],
    "codex_colon": [
        "aDefensin5.ome.tif",
        "aSMA.ome.tif",
        "BCL2.ome.tif",
        "CD117.ome.tif",
        "CD11c.ome.tif",
        "CD123.ome.tif",
        "CD127.ome.tif",
        "CD138.ome.tif",
        "CD15.ome.tif",
        "CD16.ome.tif",
        "CD161.ome.tif",
        "CD163.ome.tif",
        "CD19.ome.tif",
        "CD206.ome.tif",
        "CD21.ome.tif",
        "CD25.ome.tif",
        "CD3.ome.tif",
        "CD31.ome.tif",
        "CD34.ome.tif",
        "CD36.ome.tif",
        "CD38.ome.tif",
        "CD4.ome.tif",
        "CD44.ome.tif",
        "CD45.ome.tif",
        "CD45RO.ome.tif",
        "CD49a.ome.tif",
        "CD49f.ome.tif",
        "CD56.ome.tif",
        "CD57.ome.tif",
        "CD66.ome.tif",
        "CD68.ome.tif",
        "CD69.ome.tif",
        "CD7.ome.tif",
        "CD8.ome.tif",
        "CD90.ome.tif",
        "CollIV.ome.tif",
        "Cytokeratin.ome.tif",
        "DRAQ5.ome.tif",
        "FAP.ome.tif",
        "HLADR.ome.tif",
        "Ki67.ome.tif",
        "MUC1.ome.tif",
        "MUC2.ome.tif",
        "MUC6.ome.tif",
        "NKG2D.ome.tif",
        "OLFM4.ome.tif",
        "Podoplanin.ome.tif",
        "SOX9.ome.tif",
        "Synaptophysin.ome.tif",
        "Vimentin.ome.tif",
    ],
}

dataset_paths = {
    "mibi_breast": os.path.join(BASE_DIR, "mibi_breast", "image_data", "samples"),
    "mibi_decidua": os.path.join(BASE_DIR, "mibi_decidua", "image_data"),
    "vectra_colon": os.path.join(BASE_DIR, "vectra_colon", "raw_structured"),
    "vectra_pancreas": os.path.join(BASE_DIR, "vectra_pancreas", "raw_structured"),
    "codex_colon": os.path.join(BASE_DIR, "codex_colon", "raw_structured"),
}


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def change_lmdb_envs(
    dataset_lmdb_dir,
    file_idx,
    env_imgs=None,
    env_labels=None,
    env_metadata=None,
):
    if env_imgs is not None:
        env_imgs.close()
        env_labels.close()
        env_metadata.close()

    lmdb_imgs_path = os.path.join(dataset_lmdb_dir, str(file_idx) + "-TRAIN_images")
    lmdb_labels_path = os.path.join(dataset_lmdb_dir, str(file_idx) + "-TRAIN_labels")
    lmdb_metadata_path = os.path.join(dataset_lmdb_dir, str(file_idx) + "-TRAIN_metadata")
    os.makedirs(lmdb_imgs_path, exist_ok=True)
    os.makedirs(lmdb_labels_path, exist_ok=True)
    os.makedirs(lmdb_metadata_path, exist_ok=True)

    env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
    env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_IMG)
    env_metadata = lmdb.open(lmdb_metadata_path, map_size=MAP_SIZE_META)

    txn_meta, txn_imgs, txn_labels = (
        env_metadata.begin(write=True),
        env_imgs.begin(write=True),
        env_labels.begin(write=True),
    )
    return txn_meta, txn_imgs, txn_labels


def load_channel_img(channel_path):
    channel_img = iio.imread(channel_path)  # (N M)
    channel_img = normalize(np.squeeze(channel_img))
    channel_img = (channel_img * 255).astype(np.uint8)
    return channel_img


def main(args):
    start_fov_idx = args.start_fov_idx
    end_fov_idx = args.end_fov_idx

    patch_size = args.patch_size
    sel_dataset_paths = {k: dataset_paths[k] for k in args.dataset_keys}

    base_lmdb_dir = BASE_DIR + args.base_lmdb_dir_name
    os.makedirs(base_lmdb_dir, exist_ok=True)

    for dataset, path in sel_dataset_paths.items():
        print(f"PROCESSING DATASET {dataset} stored in {path}...")
        dataset_lmdb_dir = os.path.join(base_lmdb_dir, dataset)
        file_idx = 0
        env_imgs, env_labels, env_metadata = (
            None,
            None,
            None,
        )

        fovs = os.listdir(path)[start_fov_idx:end_fov_idx]
        print(f"TOTAL #FOVS {len(fovs)} FOR DATASET {dataset}")

        lmdb_imgs_path = os.path.join(
            dataset_lmdb_dir,
            str(file_idx) + "-TRAIN_images",
        )
        lmdb_labels_path = os.path.join(
            dataset_lmdb_dir,
            str(file_idx) + "-TRAIN_labels",
        )
        lmdb_metadata_path = os.path.join(
            dataset_lmdb_dir,
            str(file_idx) + "-TRAIN_metadata",
        )
        os.makedirs(lmdb_imgs_path, exist_ok=True)
        os.makedirs(lmdb_labels_path, exist_ok=True)
        os.makedirs(lmdb_metadata_path, exist_ok=True)

        env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
        env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_IMG)
        env_metadata = lmdb.open(lmdb_metadata_path, map_size=MAP_SIZE_META)

        naming_convention = naming_convention_dict[dataset]
        with (
            env_metadata.begin(write=True) as txn_meta,
            env_imgs.begin(write=True) as txn_imgs,
            env_labels.begin(write=True) as txn_labels,
        ):
            for img_idx, fov in tqdm(enumerate(sorted(fovs)), total=len(fovs)):
                fov_name_cleaned = "".join(e for e in str(fov) if e.isalnum())
                do_print = img_idx % 50 == 0
                if do_print:
                    print(f'idx: {img_idx}/{len(fovs)}, fov: "{fov_name_cleaned}"')

                img_idx = f"{dataset}_{img_idx:04d}"
                metadata_dict = {}

                fov_path = os.path.join(path, fov)
                channels = selected_channels[dataset]
                channel_names = [channel.split(".")[0] for channel in channels]
                channel_paths = [os.path.join(fov_path, channel) for channel in channels]

                # get metadata
                metadata_dict["fov"] = fov
                metadata_dict["channel_names"] = channel_names
                metadata_bytes = json.dumps(metadata_dict).encode("utf-8")

                # get segmentation mask
                segmentation_path = naming_convention(fov)
                # segmentation mask has to be uint16 because of values of to ~3000 segments
                # Thus, cannot be jpeg compressed
                segmentation_mask = iio.imread(segmentation_path).squeeze().astype(np.uint16)

                for ch_idx, channel_path in enumerate(channel_paths):
                    channel_img = load_channel_img(channel_path)
                    # do crops
                    x_crop_idx = y_crop_idx = 0
                    x_reached_end = y_reached_end = False
                    while patch_size * x_crop_idx < channel_img.shape[0] and not x_reached_end:
                        x_0 = patch_size * x_crop_idx
                        x_1 = patch_size * (x_crop_idx + 1)
                        if channel_img.shape[0] - x_1 < patch_size / 2:
                            # if less than half a patch remains, we take it all
                            x_1 = -1
                            x_reached_end = True

                        while patch_size * y_crop_idx < channel_img.shape[1] and not y_reached_end:
                            y_0 = patch_size * y_crop_idx
                            y_1 = patch_size * (y_crop_idx + 1)
                            if channel_img.shape[1] - y_1 < patch_size / 2:
                                y_1 = -1
                                x_reached_end = True

                            crop = channel_img[x_0:x_1, y_0:y_1]
                            crop_mask = segmentation_mask[x_0:x_1, y_0:y_1]
                            crop_jpg_encoded = iio.imwrite(
                                "<bytes>",
                                crop,
                                extension=".jpeg",
                            )
                            patch_idx = f"{img_idx}_p{x_crop_idx + y_crop_idx}"
                            crop_ch_idx_bytes = f"{patch_idx}_ch{ch_idx}".encode("utf-8")
                            txn_imgs.put(
                                crop_ch_idx_bytes,
                                crop_jpg_encoded,
                            )

                            patch_idx_bytes = patch_idx.encode("utf-8")
                            txn_labels.put(
                                patch_idx_bytes,
                                crop_mask.tobytes(),
                            )
                            txn_meta.put(
                                patch_idx_bytes,
                                metadata_bytes,
                            )

                            y_crop_idx += 1
                        x_crop_idx += 1

            """
            # save patch, label, fov, dataset and channel_names for each training sample
            print(f"Saving {len(patches)} patches for img {img_idx}")
            for p_idx, patch in enumerate(patches):
                patch_bytes = patch.tobytes()
                full_idx = f"{img_idx}_{p_idx:03d}"

                idx_bytes = str(full_idx).encode("utf-8")
                txn_imgs.put(idx_bytes, patch_bytes)
            """

        env_imgs.close()
        env_metadata.close()
        env_labels.close()
        print(f"FINISHED DATASET {dataset}, SAVED AT: {dataset_lmdb_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        help="Patch size",
        default=512,
    )
    parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        type=int,
        help="Number of jobs to run in parallel",
        default=8,
    )
    parser.add_argument(
        "--do_test_run",
        action=argparse.BooleanOptionalAction,
        help="Toggle test run with small subset of dataset",
        default=False,
    )
    parser.add_argument(
        "--dataset_keys",
        nargs="+",
        help="""Names of datasets to process. One or more of the follwowing:
        mibi_breast, mibi_decidua, vectra_colon, vectra_pancreas, codex_colon""",
    )
    parser.add_argument(
        "--start_fov_idx",
        type=int,
        help="Start index of FOVs to process",
        default=0,
    )
    parser.add_argument(
        "--end_fov_idx",
        type=int,
        help="End index of FOVs to process",
        default=-1,
    )
    parser.add_argument(
        "--do_cell_crops",
        action=argparse.BooleanOptionalAction,
        help="Toggle cell crops generation, otherwise whole multiplex images are saved",
        default=False,
    )
    parser.add_argument(
        "--base_lmdb_dir_name",
        type=str,
        help="Base lmdb dir name",
        default="_lmdb",
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))


# channel_imgs = Parallel(n_jobs=n_jobs)(
#    delayed(load_channel)(channel_path) for channel_path in channel_paths
# )
