# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from torch.nn.utils.rnn import pad_sequence

from dinov2.layers import (
    MemEffAttention,
    Mlp,
    PatchEmbed,
    PatchEmbedPerChannel,
    SwiGLUFFNFused,
)
from dinov2.layers import (
    NestedTensorBlock as Block,
)
from dinov2.utils.utils import exists

logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, attn_mask=None, local_crop_len=None):
        for b in self:
            x = b(x, attn_mask, local_crop_len)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        free_shapes=None,
        num_loc_crops=8,
        use_ch_patch_embed=False,
        gray_scale=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.img_size = img_size
        self.in_chans = in_chans
        self.gray_scale = gray_scale


        self.use_ch_patch_embed = use_ch_patch_embed
        if self.use_ch_patch_embed:
            print(f"---- Using PatchEmbedPerChannel, with {in_chans} channels ----")
            embed_layer = PatchEmbedPerChannel
        else:
            print(f"---- Using PatchEmbed, with {in_chans} channels ----")
            embed_layer = PatchEmbed

        if isinstance(in_chans, int):
            self.patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                gray_scale=gray_scale,
            )
            num_patches = self.patch_embed.num_patches
        else:  # list of channels
            self.patch_embed = {
                ch: embed_layer(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=ch,
                    embed_dim=embed_dim,
                )
                for ch in in_chans
            }
            num_patches = self.patch_embed[min(in_chans)].num_patches

        if self.use_ch_patch_embed:
            num_patches = int(num_patches / in_chans)
        self.num_loc_crops = num_loc_crops
        self.free_shapes = free_shapes

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolates the positional encoding of the given input tensor based on its shape and the provided width and height.

        Args:
            x (Tensor): The input tensor.
            w (int): The width of the input tensor.
            h (int): The height of the input tensor.

        Returns:
            Tensor: The interpolated positional encoding tensor.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        if self.use_ch_patch_embed:
            patch_pos_embed = patch_pos_embed.expand(1, self.in_chans, -1, dim).reshape(
                1, -1, dim
            )
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def interpolate_pos_encoding_lc_navit(self, x, crop_dims, crop_nb_list):
        """
        Interpolates position encoding for local crops of an image.

        Args:
            x: The input tensor. b n d
            crop_dims: A list of crop dimensions. b [n_c, 2]
            crop_nb_list: A list of crop numbers. b n_c [n_p]

        Returns:
            A tensor of interpolated position encodings.
        """
        previous_dtype = x.dtype
        N_ref = self.pos_embed.shape[1] - 1
        dim = self.pos_embed.shape[-1]  # 384
        max_nb_token = x.shape[1]
        # print(f"max_nb_token = {max_nb_token}")

        M = int(math.sqrt(N_ref))  # Recover the number of patches in each dimension
        assert N_ref == M * M
        # x.shape = (b, n, d), pos_embed.shape = (1, n_ref, d)

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        batch_embeds_list = []
        for crops_dim, crops_nbs in zip(crop_dims, crop_nb_list):
            crop_pos_embed_list = []
            for single_dim, nbs in zip(crops_dim, crops_nbs):
                w, h = int(single_dim[0]), int(single_dim[1])
                w0 = w // self.patch_size
                h0 = h // self.patch_size

                crop_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                    mode="bicubic",
                    antialias=self.interpolate_antialias,
                    size=(w0, h0),
                )
                assert (w0, h0) == crop_pos_embed.shape[-2:]
                crop_pos_embed = crop_pos_embed.permute(0, 2, 3, 1).view(
                    -1, dim
                )  # n_t dim
                # keep only pos_embeds of selected patches
                # print(f"nbs  {nbs.device} {nbs.dtype} {nbs[:10]}") # FIXME: sometimes dtype is float16?
                crop_pos_embed = torch.index_select(
                    crop_pos_embed,
                    dim=0,
                    index=nbs.long(),
                )  # n_p_select d

                crop_pos_embed_list.append(
                    torch.cat([class_pos_embed, crop_pos_embed], dim=0)
                )

            # pad with zeros
            all_crops_embed = torch.cat(crop_pos_embed_list, dim=0)
            padding = torch.zeros(
                (
                    max_nb_token - all_crops_embed.shape[0],
                    dim,
                ),
                dtype=all_crops_embed.dtype,
                device=all_crops_embed.device,
            )

            batch_embeds_list.append(
                torch.cat([all_crops_embed, padding], dim=0)
            )  # cat on tokens nb
        return (
            torch.stack(batch_embeds_list)
            .to(torch.cuda.current_device())
            .to(previous_dtype)
        )

    def prepare_tokens_with_masks(
        self,
        x: torch.Tensor,  # (b n d)
        masks: Optional[torch.Tensor] = None,  # (b d)
        local_patch_pos: Optional[List[List[int]]] = None,
        local_crop_dims: Optional[List[List[int]]] = None,
        local_crop_len: List[torch.Tensor] = None,
        num_ch_list: List[int] = None,
    ) -> torch.Tensor:
        """Prepare tokens with positional embeddings and masks.

        Args:
            x (torch.Tensor): input image, shape = (b, c, w, h)
            masks (Optional[torch.Tensor], optional): masks to apply on input image. Defaults to None.
            free_shapes (bool, optional): whether or not the input image shapes are variable.
                Defaults to False.
            local_patch_pos (Optional[List[List[int]]], optional): list of patch positions
                for each image in the batch. Only used when free_shapes is True.
                Defaults to None.
            local_crop_dims (Optional[List[List[int]]], optional): list of image shapes
                for each image in the batch. Only used when free_shapes is True.
                Defaults to None.
        Note: local_patch_pos and local_crop_dims are None for Global crops

        Returns:
            torch.Tensor: tokens with positional embeddings, shape = (b, n, d)
        """
        """
        print(
            "prepare_tokens_with_masks x.shape local_patch_pos local_crop_dims local_crop_len num_ch_list"
        )
        print(
            f"prepare_tokens_with_masks {x.shape} {local_patch_pos} {local_crop_dims} {local_crop_len} {num_ch_list}"
        )"""

        # newly created pos embed vect also needs padding
        # b c w h OR b c p (n p)
        # TODO: Problem: is using np p as w h --> aspect ratio severly distorted....
        b, c, w, h = x.size()
        print(f"prepare_tokens_with_masks x.shape 1 {x.shape}")
        if isinstance(self.patch_embed, dict):  # here we have one tokenizer for each nb channels
            x = self.patch_embed[c](x)
            print(f"patch embed x.shape 1 {x.shape}")
        else:
            x = self.patch_embed(x)  # b n d (=384)
            print(f"patch embed x.shape 2 {x.shape}")
            if num_ch_list is not None:  # means that x = (b c) n d
                b = len(num_ch_list)
                x_list = torch.split(x, num_ch_list, dim=0)  # b [c n d]
                x = pad_sequence(x_list, batch_first=True)  # b c_max n d
                x = x.reshape(b, -1, x.shape[-1])  # b (c_max n) d

        x_dim = x.shape[-1]
        print(f"prepare_tokens_with_masks x.shape 2 {x.shape}")
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
        print(f"prepare_tokens_with_masks x.shape 3 {x.shape}")
        if (
            self.free_shapes and local_patch_pos is not None
        ):  # if local crops w free shape
            cls_token = torch.tile(
                self.cls_token, dims=(x.shape[0], self.num_loc_crops, 1)
            )  # (1 1 d) -> (b n_lc d)
            x_list = []
            local_crop_len = [
                torch.cat([torch.zeros(1, device=el.device), el])
                for el in local_crop_len
            ]
            max_nb_token = x.shape[1] + self.num_loc_crops  # (with cls tokens)
            for i in range(x.shape[0]):  # b n d
                single_x_list = []
                for j in range(1, len(local_crop_len[i])):
                    # print(
                    #    f"""i {i} j {j} {int(local_crop_len[i][:j].sum())}
                    #     {int(local_crop_len[i][:j + 1].sum())} {local_crop_len[i][:j+1]} {len(local_crop_len[i])}"""
                    # add padding here to even up x, or make x a nested tensor
                    single_x_list.append(
                        torch.cat(
                            (
                                cls_token[i, j - 1, :][None, None, :],
                                x[
                                    i,
                                    int(local_crop_len[i][:j].sum()) : int(
                                        local_crop_len[i][: j + 1].sum()
                                    ),
                                    :,
                                ][None, :],
                            ),
                            dim=1,
                        )
                    )
                cat_x = torch.cat(single_x_list, dim=1)
                padding = torch.zeros(
                    (1, max_nb_token - cat_x.shape[1], x_dim),
                    device=cat_x.device,
                    dtype=cat_x.dtype,
                )
                cat_x = torch.cat([cat_x, padding], dim=1)
                x_list.append(cat_x)
            # x = torch.nested.nested_tensor(x_list)  # local_patch_pos b n d
            x = torch.cat(x_list, dim=0)  # local_patch_pos b n d

            interpolated_pos_embeds = self.interpolate_pos_encoding_lc_navit(
                x,
                crop_dims=local_crop_dims,
                crop_nb_list=local_patch_pos,
            )

        else:
            print(f"prepare_tokens_with_masks x.shape {x.shape}")
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            interpolated_pos_embeds = self.interpolate_pos_encoding(x, w, h)

            if (
                self.use_ch_patch_embed
                and interpolated_pos_embeds.shape[1:] != x.shape[1:]
            ):
                if exists(num_ch_list):
                    c_max = max(num_ch_list)
                    pos_embeds = []
                    for ch_nb in num_ch_list:
                        allch_pos_embeds = interpolated_pos_embeds[:, 1:, :].tile(
                            1, ch_nb, 1
                        )
                        pos_embed_padding = torch.zeros(
                            (
                                1,
                                (c_max - ch_nb)
                                * (interpolated_pos_embeds.shape[1] - 1),
                                interpolated_pos_embeds.shape[-1],
                            ),
                            device=interpolated_pos_embeds.device,
                            dtype=interpolated_pos_embeds.dtype,
                        )
                        allch_pos_embeds = torch.cat(
                            (
                                interpolated_pos_embeds[:, :1, :],
                                allch_pos_embeds,
                                pos_embed_padding,
                            ),
                            dim=1,
                        )
                        pos_embeds.append(allch_pos_embeds)

                    interpolated_pos_embeds = torch.cat(pos_embeds, dim=0)
                else:
                    # -1 to account for cls token
                    all_pos_embeds = interpolated_pos_embeds[:, 1:, :].tile(1, c, 1)
                    interpolated_pos_embeds = torch.cat(
                        (
                            interpolated_pos_embeds[:, :1, :],
                            all_pos_embeds,
                        ),
                        dim=1,
                    )

        x = x + interpolated_pos_embeds
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, 0],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(
        self,
        x_list,
        masks_list,
        attn_mask_list=None,
        local_crop_len=None,
        local_patch_pos=None,
        local_crop_dims=None,
        num_ch_list=None,
    ):
        """
        Forward pass for a list of input features with masks.
        Args:
            x_list: List of input features.
            masks_list: List of masks corresponding to the input features.
            attn_mask_list: Optional list of attention masks.
            local_crop_len: Length of local crop.
            local_patch_pos: Position of local patches.
            local_crop_dims: Dimensions of local crops.

        Returns:
            List of dictionaries containing normalized input features and masks.
        """
        if exists(local_patch_pos) and exists(local_crop_dims):
            x = [
                self.prepare_tokens_with_masks(
                    x,
                    masks,
                    local_patch_pos=patch_pos,
                    local_crop_dims=crop_dims,
                    local_crop_len=crop_len,
                )
                for x, masks, patch_pos, crop_dims, crop_len in zip(
                    x_list,
                    masks_list,
                    local_patch_pos,
                    local_crop_dims,
                    local_crop_len,
                )
            ]
        elif exists(num_ch_list):
            x = [
                self.prepare_tokens_with_masks(x, masks, num_ch_list=ch_list)
                for x, masks, ch_list in zip(x_list, masks_list, num_ch_list)
            ]
        else:
            x = [
                self.prepare_tokens_with_masks(x, masks)
                for x, masks in zip(x_list, masks_list)
            ]
        # x_list = [global_crops, local_crops]
        # masks_list = [masks, None]
        # x[0] = B C H W
        # x[1] = B N D
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask_list, local_crop_len=local_crop_len)

        all_x = x
        output = []
        for i, (x, masks) in enumerate(zip(all_x, masks_list)):
            x_norm = self.norm(x)
            if (
                self.free_shapes
                and local_patch_pos is not None
                and local_patch_pos[i] is not None
            ):
                n_cls_tokens = self.num_loc_crops
            else:
                n_cls_tokens = 1
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, :n_cls_tokens],
                    "x_norm_regtokens": x_norm[
                        :, n_cls_tokens : self.num_register_tokens + 1
                    ],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def pad_token_list_to_fixed_len(self, token_list: list, masks=None):
        # token_list: B x N_var D
        token_nested_tensor = torch.nested.nested_tensor(token_list)
        token_padded_tensor = torch.nested.to_padded_tensor(token_nested_tensor, 0.0)
        # mask added pads --> TODO: Review this as pads could be used as registers
        if masks is not None:
            token_padded_tensor = torch.where(
                token_padded_tensor == 0.0,
                self.mask_token.to(token_padded_tensor.dtype).unsqueeze(0),
                token_padded_tensor,
            )
        return token_padded_tensor

    def pad_token_tensor_to_fixed_len(self, token_tensor: torch.Tensor):
        # Useless bc Tensor has regular dimensions??
        b, curr_n, d = token_tensor.shape
        # token_tensor: B N D
        n_pad_tokens = self.num_tokens - curr_n

        pad_tokens = torch.ones(b, n_pad_tokens, d)
        return torch.cat([token_tensor, pad_tokens], dim=1)

    def forward_features(
        self,
        x,
        masks=None,
        attn_masks=None,
        local_crop_len=None,
        local_patch_pos=None,
        local_crop_dims=None,
        num_ch_list=None,
    ):
        if isinstance(x, list):
            return self.forward_features_list(
                x,
                masks,
                attn_masks,
                local_crop_len=local_crop_len,
                local_patch_pos=local_patch_pos,
                local_crop_dims=local_crop_dims,
                num_ch_list=num_ch_list,
            )

        # if not list, we only have gc, hence no local_patch_pos or local_crop_dims
        x = self.prepare_tokens_with_masks(x, masks, num_ch_list=num_ch_list)

        """ # Already done in collate
        if x.shape[1] < self.num_tokens:
            # Add padding tokens to reach fixed len
            x = self.pad_token_tensor_to_fixed_len(x)
        """

        for blk in self.blocks:
            x = blk(x, attn_masks)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [
                self.norm(out) for out in outputs
            ]  # mem leak occurs here if mlp commented
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def count_parameters(model: nn.Module, with_grad: bool = True):
    if with_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
