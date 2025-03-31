#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

# For type hints
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV2ImageEncoder
from .coordinate_encoder_model import SAMV2CoordinateEncoder
from .prompt_encoder_model import SAMV2PromptEncoder
from .mask_decoder_model import SAMV2MaskDecoder
from .memory_encoder_model import SAMV2MemoryEncoder
from .memory_fusion_model import SAMV2MemoryFusion


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2Model(nn.Module):
    """
    Wrapper around separated SAMV2 model components, so that the model can be used as a singular entity
    """

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        memory_encoder_model: SAMV2MemoryEncoder,
        memory_fusion_model: SAMV2MemoryFusion,
    ):

        # Inherit from parent
        super().__init__()

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self.memory_encoder = memory_encoder_model
        self.memory_fusion = memory_fusion_model

        # Store 'empty' prompt encoding, used when stepping video (so we don't repeatedly re-compute it!)
        no_prompt_encoding = self.encode_prompts([], [], [])
        self.register_buffer("_video_noprompt_encoding", no_prompt_encoding, persistent=False)

        # Default to eval mode, expecting to use inference only
        self.eval()

    # .................................................................................................................

    def set_window_size(self, window_size: int | None):
        """
        Function used to adjust image encoder window sizing.
        This function doesn't do anything for SAM v2, which has varying
        window sizes per hierarchical stage. It is just here for compatibility
        with the V1 model, which does support dynamic size adjustments
        """

        return self

    # .................................................................................................................

    def forward(
        self,
        image_rgb_normalized_bchw: Tensor,
        boxes_tensor: Tensor,
        fg_tensor: Tensor,
        bg_tensor: Tensor,
        mask_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        All image/mask generating code of SAMV2 model, bundled into a single function.
        Takes an image and set of prompts and produces several candidate segmentation masks.

        Note that in practice, it makes more sense to call the component pieces of the model,
        rather than using this function so that image & prompt encoding can happen independently.
        See the 'encode_prompts', 'encode_image' and 'generate_masks' functions for more info

        Returns:
            mask_predictions, iou_predictions, objscore_pred, mask_tokens_out
        """

        # Encode prompts & image inputs
        box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
        encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)
        encoded_image = self.image_encoder(image_rgb_normalized_bchw)

        # Combine encodings to generate mask output
        patch_grid_hw = encoded_image.shape[2:]
        grid_posenc = self.coordinate_encoder.get_full_grid_encoding(patch_grid_hw)
        mask_preds, iou_preds, objscore_pred, mask_tokens_out = self.mask_decoder(
            encoded_image, encoded_prompts, grid_posenc, mask_hint
        )

        return mask_preds, iou_preds, objscore_pred, mask_tokens_out

    # .................................................................................................................

    def encode_prompts(self, box_tlbr_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
        """
        Function used to encode prompt coordinates. Inputs should be given as lists
        of prompts. The length of each list does not need to match. Enter either
        None or an empty list ([]) to disable any of the prompts.

        Box prompt formatting:
            Each entry should be in top-left/bottom-right form: ((x1, y1), (x2, y2))
            For example:
                [
                    [(0.1, 0.5), (0.3, 0.7)], # Box 1
                    [(0.6, 0.2), (0.8, 0.4)], # Box 2
                    ... etc ...
                ]

        FG/BG prompt formatting:
            Each entry should be a single (x, y) point
            For example:
                [
                    (0.2, 0.6), # Point 1
                    (0.5, 0.4), # Point 2
                    (0.7, 0.7), # Point 3
                    (0.1, 0.9), # Point 4
                    ... etc ..
                ]

        Returns:
            encoded_prompts (shape: 1 x N x F, where N is number of prompt points, F is features per prompt)
        """

        with torch.inference_mode():
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length=1024,
        use_square_sizing=True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:

        with torch.inference_mode():
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
            encoded_image_features_list = self.image_encoder(image_rgb_normalized_bchw)

        # Get patch sizing of lowest-res tokens (as needed by other components)
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        return encoded_image_features_list, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image_features_list: list[Tensor],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:

        # Get sizing of the lowest-resolution image encoding
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        with torch.inference_mode():
            grid_posenc = self.coordinate_encoder.get_full_grid_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                encoded_image_features_list, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return mask_preds, iou_preds

    # .................................................................................................................

    def initialize_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        box_tlbr_norm_list: list,
        fg_xy_norm_list: list,
        bg_xy_norm_list: list,
        mask_hint: Tensor | int | None = None,
        mask_index_select: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:

        # Encode initial prompts
        encoded_prompts = self.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

        with torch.inference_mode():

            # For convenience
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            token_hw = lowres_imgenc.shape[2:]

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.coordinate_encoder.get_full_grid_encoding(token_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                encoded_image_features_list,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Select mask to use for initial encodings (auto-select if not given an index)
            if mask_index_select is None:
                mask_index_select = self.mask_decoder.get_best_mask_index(iou_preds)
            best_mask_pred = mask_preds[:, [mask_index_select], :, :]
            best_obj_ptr = obj_ptrs[:, [mask_index_select]]

            # Encode initial memory
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, is_prompt_encoding=True)

        return best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def step_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        with torch.inference_mode():

            # Encode image features with previous memory encodings & object pointer data
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            memfused_encimg = self.memory_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                previous_memory_encodings,
                previous_object_pointers,
            )

            # Run mask decoder on memory encoded features
            # Called '_forward_sam_heads' in original code
            # See: https://github.com/facebookresearch/segment-anything-2/blob/6ba4c65cb2ccaff418610662eb96d3eb4a77eaf4/sam2/modeling/sam2_base.py#L762
            patch_grid_hw = memfused_encimg.shape[2:]
            grid_posenc = self.coordinate_encoder.get_full_grid_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                [memfused_encimg, *hires_imgenc],
                self._video_noprompt_encoding,
                grid_posenc,
                mask_hint=None,
                blank_promptless_output=False,
            )

            # Keep only the 'best' results
            best_mask_pred, _, best_obj_ptr = self.mask_decoder.get_best_decoder_results(
                mask_preds,
                iou_preds,
                obj_ptrs,
                exclude_0th_index=True,
            )

            # Encode new memory features
            # See: https://github.com/facebookresearch/segment-anything-2/blob/6ba4c65cb2ccaff418610662eb96d3eb4a77eaf4/sam2/modeling/sam2_base.py#L787
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred)

        return obj_score, best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def get_best_mask_index(self, iou_predictions: Tensor) -> int:
        """Returns the index of the highest IoU prediction score"""
        return self.mask_decoder.get_best_mask_index(iou_predictions)

    # .................................................................................................................
