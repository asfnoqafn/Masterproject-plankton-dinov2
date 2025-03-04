from dinov2.data.loaders import make_dataset
from dinov2.data import DataAugmentationDINO

import torch
from tqdm import tqdm

from dinov2.data import (
    DataAugmentationDINO,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
)

from timeit import default_timer as timer
import torch.nn.functional as nnf
from functools import partial




root = "/home/hk-project-p0021769/hgf_auh3910/own_data/ISIISNet/"
ds_path = f"LMDBDataset:split=TRAIN:root={root}:extra=*"


gpu_experiment = []

########################################## CPU Code ##########################################

inputs_dtype = torch.half
img_size = 224
patch_size = 14
batch_size = 256
sampler_type = SamplerType.SHARDED_INFINITE

n_tokens = (img_size // patch_size) ** 2

mask_generator = MaskingGenerator(
    input_size=(
        img_size // patch_size,
        img_size // patch_size,
    ),
    max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
)

aug_kwargs = {
    "global_crops_scale": (0.32, 1.0),
    "local_crops_scale": (0.05, 0.32),
    "local_crops_number": 8,
    "global_crops_size": 224,
    "local_crops_size": 98,
    "patch_size": 14,
    "use_native_res": False,
    "do_seg_crops": None,
    "do_multi_channel": False,
}
data_transform_cpu = DataAugmentationDINO(use_kornia=True, **aug_kwargs)

collate_fn_cpu = partial(
    collate_data_and_cast,
    mask_ratio_tuple=(0.1, 0.5),
    mask_probability=0.5,
    n_tokens=n_tokens,
    mask_generator=mask_generator,
    dtype=inputs_dtype,
    do_free_shapes=None,
    use_ch_patch_embed=False,
    use_variable_channels=False,
)

dataset = make_dataset(
    dataset_str=ds_path,
    transform=data_transform_cpu,
    target_transform=lambda _: (),
    with_targets=False,
    cache_dataset=False,
)

dl_kwargs = {
    "dataset": dataset,
    "batch_size": batch_size,
    "num_workers": 16,
    "shuffle": True,
    "seed": 0,
    "sampler_type": sampler_type,
    "sampler_advance": 0,
    "drop_last": True,
}
data_loader = make_data_loader(collate_fn=collate_fn_cpu, **dl_kwargs)

batches = 100
start = timer()
for data in tqdm(data_loader):
    if batches <= 0:
        break
    batches -= 1
end = timer()

time = end - start
gpu_experiment.append({'batch_size': batch_size, 'device': 'cpu', 'time': time})
    


print(gpu_experiment)

########################################## GPU Code ##########################################


def collate_fn_cpu(batch):
    collated = []
    for item in batch:
        image = item[0].unsqueeze(0)
        resized = nnf.interpolate(image, size=(128, 128), mode='bicubic')
        collated.append(resized)

    return torch.cat(collated, 0)

batch_size = 256

aug_kwargs = {
    "global_crops_scale": (0.32, 1.0),
    "local_crops_scale": (0.05, 0.32),
    "local_crops_number": 8,
    "global_crops_size": 224,
    "local_crops_size": 98,
    "patch_size": 14,
    "use_native_res": False,
    "do_seg_crops": None,
    "do_multi_channel": False,
}
data_transform = DataAugmentationDINO(use_kornia=True, **aug_kwargs)

dataset = make_dataset(
    dataset_str=ds_path,
    transform=None,
    target_transform=lambda _: (),
    with_targets=False,
    cache_dataset=False,
)

data_loader = make_data_loader(dataset=dataset, batch_size=batch_size, num_workers=16, collate_fn=collate_fn_cpu)

batches = 100
start = timer()
for data in tqdm(data_loader):
    data = data.to(device=f"cuda:{torch.cuda.current_device()}")
    data = data_transform(data)
    if batches <= 0:
        break
    batches -= 1
end = timer()

time = end - start
gpu_experiment.append({'batch_size': batch_size, 'device': 'gpu', 'time': time})


print(gpu_experiment)


