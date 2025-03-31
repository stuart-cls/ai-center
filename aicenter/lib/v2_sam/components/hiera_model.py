#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .hiera_blocks import PooledWindowedBlock, WindowedBlock, TransformerBlock

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HieraModel(nn.Module):
    """
    Simplified implementation of Hiera image encoder model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This model is a multi-stage vision-transformer which uses windowed attention on most blocks.
    At each stage (except the first) inputs are pooled, which results in spatial downsampling of
    processed image tokens. Some stages include equally spaced non-windowed attention blocks.

    The output of the model is a list of encoded image tokens output from each of the
    stages of the model. Each set of tokens is progressively halved in width & height,
    while doubled in feature count.

    This implementation hard-codes some of the structural patterns of the original implementation.
    Notably, this version explicitly represents the stages of the model as sub-modules.

    The original implementation can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/57bc94b7391e47e5968004a0698f8bf793a544d1/sam2/modeling/backbones/hieradet.py#L171

    The original model architecture is described in:
        "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles"
        By: Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal,
        Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer
        @ https://arxiv.org/abs/2306.00989
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token_1st_stage=96,
        num_heads_1st_stage=1,
        blocks_per_stage=(2, 3, 16, 3),
        window_size_per_stage=(8, 4, 14, 7),
        global_attention_spacing_per_stage=(None, None, 4, None),
    ):
        # Inherit from parent
        super().__init__()

        # Compute multiplier-based configs
        stage_multiplier = [2**stage_idx for stage_idx, _ in enumerate(blocks_per_stage)]
        features_per_stage = [features_per_token_1st_stage * mult for mult in stage_multiplier]
        heads_per_stage = [num_heads_1st_stage * mult for mult in stage_multiplier]

        # Generate configs that are different on the first stage
        initial_winsize_per_stage = [window_size_per_stage[0], *window_size_per_stage[:-1]]
        is_pooled_per_stage = [stage_idx > 0 for stage_idx, _ in enumerate(blocks_per_stage)]

        # Bundle per-stage config arguments and build stage modules
        stage_iter = zip(
            features_per_stage,
            heads_per_stage,
            blocks_per_stage,
            window_size_per_stage,
            initial_winsize_per_stage,
            global_attention_spacing_per_stage,
            is_pooled_per_stage,
        )
        self.stages = nn.ModuleList(HieraStage(*args) for args in stage_iter)

        # Store feature counts so sizes can be communicate to other models
        self._features_per_stage = tuple(features_per_stage)

    # .................................................................................................................

    def forward(self, patch_tokens_bhwc: Tensor) -> list[Tensor]:

        # Store intermediate results from each stage
        stage_results = []
        for stage in self.stages:
            patch_tokens_bhwc = stage(patch_tokens_bhwc)
            stage_results.append(patch_tokens_bhwc)

        # Return results with shape: BxCxHxW
        return [result.permute(0, 3, 1, 2) for result in stage_results]

    # .................................................................................................................

    def get_features_per_stage(self) -> tuple[int, int, int, int]:
        return self._features_per_stage

    # .................................................................................................................


class HieraStage(nn.Sequential):
    """
    Represents a single stage of the hierarchical image encoder (Hiera) from SAMV2

    Each stage consists of a sequence of transformer blocks for encoding image patch tokens.
    Except for the first stage (generally?) each stage begins with a 2x2 max-pooling, which
    reduces the spatial size of tokens which doubling the features per token.

    Within the 3rd stage of the model, there are always (so far?) 3 blocks which use global
    attention (i.e. not windowed), which are all equally spaced starting from the final
    block of the stage. This pattern is hard-coded into this implementation.

    Note: This module is not present in the original implementation. Instead all blocks are
    configured as a single sequence, with per-stage configurations handled on init.
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token,
        num_heads,
        num_blocks,
        window_size,
        window_size_1st_layer,
        global_attention_spacing,
        requires_first_layer_pooling,
    ):

        # Figure out global attention layer indices
        last_block_idx = num_blocks - 1
        no_global_attn = global_attention_spacing is None
        global_attn_idxs = [] if no_global_attn else [last_block_idx - k * global_attention_spacing for k in range(3)]

        # Figure out the first block of the stage, which may require pooling
        FirstBlockModule = PooledWindowedBlock if requires_first_layer_pooling else WindowedBlock
        first_block = FirstBlockModule(features_per_token, num_heads, window_size_1st_layer)

        # Build remaining blocks
        blocks_list = [first_block]
        for block_idx in range(1, num_blocks):

            # Use windowed or global attention blocks as needed
            is_global_attn_layer = block_idx in global_attn_idxs
            if is_global_attn_layer:
                block = TransformerBlock(features_per_token, num_heads)
            else:
                block = WindowedBlock(features_per_token, num_heads, window_size)
            blocks_list.append(block)

        # Inherit from parent
        super().__init__(*blocks_list)

    # .................................................................................................................
