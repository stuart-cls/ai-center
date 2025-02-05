import time
from collections import deque
from dataclasses import dataclass, field, astuple
from pathlib import Path

import cv2
import numpy
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .lib.make_sam_v2 import make_samv2_from_original_state_dict

SAM2_MODEL_LARGE = Path("/home/reads/src/segment-anything-2/checkpoints/sam2_hiera_large.pt")

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
        print(f"using device: {self.device}")

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


@dataclass
class TrackedObject:
    prompt_memory_encodings: list[torch.Tensor] = field(default_factory=list)
    prompt_object_pointers: list[torch.Tensor] = field(default_factory=list)
    prev_memory_encodings: deque[torch.Tensor] = field(default_factory=deque)
    prev_object_pointers: deque[torch.Tensor] = field(default_factory=deque)

    def __post_init__(self):
        # Set max lengths of previous knowledge
        self.prev_memory_encodings = deque([], maxlen=6)
        self.prev_object_pointers = deque([], maxlen=15)


class TrackingSAM(SAM2):
    def __init__(self, model_path: Path=SAM2_MODEL_LARGE):
        super().__init__(model_path)
        self.tracked_objects = []
        # single initial prompt to start
        self.init = False
        self.last_input_boxes = deque([], maxlen=10)

    def setup_predictor(self):
        _, sammodel = make_samv2_from_original_state_dict(str(self.model_path))
        sammodel.to(device=self.device)

        return sammodel

    def track_input_boxes(self, image: numpy.ndarray, input_boxes: numpy.ndarray, norm=None):
        if not self.init:
            self.last_input_boxes.appendleft(input_boxes)
            if (not (len(self.last_input_boxes) == self.last_input_boxes.maxlen) or
                    not all(numpy.allclose(input_boxes, a, rtol=0.1) for a in self.last_input_boxes)):
                return
            self.init = True
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
                              prompt_object_pointers=[init_ptr]))

    def predict(self, image: numpy.ndarray):
        # Select current object
        tracked_object = self.tracked_objects[0]
        # Process video frames with model
        t1 = time.perf_counter()
        encoded_imgs_list, _, _ = self.predictor.encode_image(image)
        obj_score, mask_pred, mem_enc, obj_ptr = self.predictor.step_video_masking(
            encoded_imgs_list, *astuple(tracked_object),
        )
        t2 = time.perf_counter()
        print(f"Took {round(1000 * (t2 - t1))} ms")

        # Store object results for future frames
        if obj_score < 0:
            print("Bad object score! Implies broken tracking!")
            # Drop tracked object here?
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

    def show_mask(image, mask, random_color=False, borders=True, centroid=True, bbox=True):
        if random_color:
            rng = numpy.random.default_rng()
            color = rng.integers(0, 255, size=3, dtype=numpy.uint8)
        else:
            color = numpy.array([30 / 255, 144 / 255, 255 / 255], dtype=numpy.uint8)
        h, w = mask.shape[-2:]
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
                    print(f"Centroid: {x_centroid}, {y_centroid}")
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
            image = show_mask(image, mask.squeeze(0), random_color=False, borders=False)
    return image