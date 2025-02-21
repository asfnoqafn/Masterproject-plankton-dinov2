import torch
import os

os.environ['TORCH_HOME']= '/home/hk-project-p0021769/hgf_auh3910/checkpoints/'

dinov2_vits14_reg_lc = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vits14_reg',
)
