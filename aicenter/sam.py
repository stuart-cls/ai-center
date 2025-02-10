import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, astuple
from functools import partial
from pathlib import Path

import cv2
import numpy
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from aicenter.log import get_module_logger
from aicenter.net import Result

from .lib.make_sam_v2 import make_samv2_from_original_state_dict

logger = get_module_logger(__name__)

SAM2_MODEL_LARGE = Path("/home/reads/src/segment-anything-2/checkpoints/sam2_hiera_large.pt")

@dataclass
class MaskResult(Result):
    mask: numpy.ndarray = None
    contours: numpy.ndarray = None

class SAM2:
    def __init__(self, model_path: Path=SAM2_MODEL_LARGE):
        self.model_path = model_path
        self.setup_device()
        self.predictor = self.setup_predictor()

    def setup_device(self):
        # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.debug(f"Using device: {self.device}")

        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def setup_predictor(self):
        model_cfg = "sam2_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, self.model_path, device=self.device)

        return SAM2ImagePredictor(sam2_model)

    def predict_input_boxes(self, image: numpy.ndarray, input_boxes: numpy.ndarray):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        return masks, scores

    @staticmethod
    def process_results(masks, scores, label):
        # TODO this should probably return the same type of thing as Net.process_results
        results = []
        # if only one mask result, not enough dimensions
        masks = numpy.array(masks, ndmin=4, copy=False)
        if masks.size:
            for mask, score in zip(masks, scores):
                mask = mask.squeeze(0)
                score = score.item()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_contour = max(contours, key=cv2.contourArea) if len(contours) > 0 else None
                if max_contour is not None:
                    # Calculate image moments of the detected contour
                    moments = cv2.moments(max_contour)
                    try:
                        x_centroid = round(moments['m10'] / moments['m00'])
                        y_centroid = round(moments['m01'] / moments['m00'])
                    except ZeroDivisionError:
                        x_centroid = None
                        y_centroid = None
                    else:
                        logger.debug(f"Segmentation mask centroid: {x_centroid}, {y_centroid}")
                    x, y, w, h = cv2.boundingRect(max_contour)
                    logger.debug(f'{label} found at: {x} {y} [{w} {h}], prob={score:.2f}')
                    results.append(MaskResult(label, x, y, w, h, score, x_centroid, y_centroid, mask, contours))
        return results

@dataclass
class TrackedObject:
    prompt_memory_encodings: list[torch.Tensor] = field(default_factory=list)
    prompt_object_pointers: list[torch.Tensor] = field(default_factory=list)
    prev_memory_encodings: deque[torch.Tensor] = field(default_factory=deque)
    prev_object_pointers: deque[torch.Tensor] = field(default_factory=deque)
    label: str = None

    def __post_init__(self):
        # Set max lengths of previous knowledge
        self.prev_memory_encodings = deque([], maxlen=6)
        self.prev_object_pointers = deque([], maxlen=15)

    @property
    def video_masking_inputs(self):
        return self.prompt_memory_encodings, self.prompt_object_pointers, self.prev_memory_encodings, self.prev_object_pointers


class TrackingSAM(SAM2):
    def __init__(self, model_path: Path=SAM2_MODEL_LARGE):
        super().__init__(model_path)
        self.tracked_objects = []
        # single initial prompt to start
        self.loop = False
        self.last_input_boxes = defaultdict(partial(deque, maxlen=10))

    def setup_predictor(self):
        _, sammodel = make_samv2_from_original_state_dict(str(self.model_path))
        sammodel.to(device=self.device)

        return sammodel

    def track_input_boxes(self, image: numpy.ndarray, input_boxes: numpy.ndarray, norm=None, label="default"):
        last_input_boxes = self.last_input_boxes[label]
        last_input_boxes.appendleft(input_boxes)
        if (not (len(last_input_boxes) == last_input_boxes.maxlen) or
                not all(numpy.allclose(input_boxes, a, rtol=0.1) for a in last_input_boxes)):
            return
        if label == 'loop':
            self.loop = True
        # TODO multiple boxes
        if norm is not None:
            input_boxes = input_boxes / norm
        input_boxes = input_boxes.reshape(-1, 2, 2)

        init_encoded_img, _, _ = self.predictor.encode_image(image)
        init_mask, init_mem, init_ptr = self.predictor.initialize_video_masking(
            init_encoded_img, input_boxes, [], []
        )
        self.tracked_objects.append(
            TrackedObject(prompt_memory_encodings=[init_mem],
                          prompt_object_pointers=[init_ptr],
                          label=label,
                          ))
        logger.debug(f"Added new tracked {label} object, {len(self.tracked_objects)} total")

    def track_objects(self, image: numpy.ndarray, results: dict[list[Result]], width, height):
        norm = numpy.array([width, height, width, height])
        for label, objects in results.items():
            if label == 'loop' and objects and self.predictor and not self.loop:
                # Only use the highest-scoring loop as the prompt
                loop = objects[0]
                xyxy = [loop.x, loop.y, loop.x + loop.w, loop.y + loop.h]
                input_boxes = numpy.atleast_2d(numpy.array(xyxy))
                self.track_input_boxes(image, input_boxes, norm, label)

    def predict(self, image: numpy.ndarray):
        # Select current object
        tracked_object = self.tracked_objects[0]
        # Process video frames with model
        t1 = time.perf_counter()
        encoded_imgs_list, _, _ = self.predictor.encode_image(image)
        obj_score, mask_pred, mem_enc, obj_ptr = self.predictor.step_video_masking(
            encoded_imgs_list, *tracked_object.video_masking_inputs,
        )
        t2 = time.perf_counter()
        logger.debug(f"Inference took {round(1000 * (t2 - t1))} ms, score={obj_score[0][0]:.2f}")

        # Store object results for future frames
        if obj_score < 0:
            # TODO Need to keep object for occlusion support, maybe drop when scores stay low for a long time
            logger.debug("Bad object score! Implies broken tracking! Dropping tracked object.")
            self.tracked_objects.remove(tracked_object)
            self.init = False
            return [], []
        tracked_object.prev_memory_encodings.appendleft(mem_enc)
        tracked_object.prev_object_pointers.appendleft(obj_ptr)

        # Create mask for display
        dispres_mask = torch.nn.functional.interpolate(
            mask_pred,
            size=image.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        disp_mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()

        return disp_mask, obj_score


def show_masks(image, masks):

    def show_mask_from_predict(image, mask, random_color=False, borders=True, centroid=True, bbox=True):
        if random_color:
            rng = numpy.random.default_rng()
            color = rng.integers(0, 255, size=3, dtype=numpy.uint8)
        else:
            color = numpy.array([30 / 255, 144 / 255, 255 / 255], dtype=numpy.uint8)
        h, w = mask.shape[-2:]
        print(image.shape, mask.shape, mask.reshape(h, w, 1).shape, numpy.atleast_3d(mask).shape)
        mask = mask.astype(numpy.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders or centroid or bbox:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_contour = max(contours, key=cv2.contourArea) if len(contours) > 0 else None
            if borders:
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                mask_image = cv2.drawContours(mask_image, contours, -1, (0, 255, 0, 0.5), thickness=2)
            if centroid and max_contour is not None:
                # Calculate image moments of the detected contour
                moments = cv2.moments(max_contour)
                try:
                    x_centroid = round(moments['m10'] / moments['m00'])
                    y_centroid = round(moments['m01'] / moments['m00'])
                except ZeroDivisionError:
                    pass
                else:
                    logger.debug(f"Segmentation mask centroid: {x_centroid}, {y_centroid}")
                    # Draw a marker centered at centroid coordinates
                    image = cv2.drawMarker(image, (x_centroid, y_centroid),(255, 0, 0, 1), thickness=1, markerSize=20)
            if bbox and max_contour is not None:
                rect = cv2.boundingRect(max_contour)
                image = cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0, 1), thickness=1)
        return cv2.addWeighted(image, 1, mask_image, 1, 0)

    # if only one result, not enough dimensions
    masks = numpy.array(masks, ndmin=4, copy=False)
    if masks.size:
        for mask in masks:
            image = show_mask_from_predict(image, mask.squeeze(0), random_color=False, borders=True)
    return image

def show_mask_from_result(image, result: MaskResult, random_color=False, borders=True, centroid=True, bbox=False):
    if random_color:
        rng = numpy.random.default_rng()
        color = rng.integers(0, 255, size=3, dtype=numpy.uint8)
    else:
        color = numpy.array([30 / 255, 144 / 255, 255 / 255], dtype=numpy.uint8)
    mask = result.mask
    contours = result.contours
    h, w = mask.shape[-2:]
    mask = mask.astype(numpy.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders or centroid or bbox:
        if borders:
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (0, 255, 0, 0.5), thickness=2)
        if centroid:
                image = cv2.drawMarker(image, (result.cx, result.cy),(255, 0, 0, 1), thickness=1, markerSize=20)
        if bbox:
            image = cv2.rectangle(image, (result.x, result.y), (result.x + result.w, result.y + result.h), (0, 0, 0, 1), thickness=1)
    return cv2.addWeighted(image, 1, mask_image, 1, 0)