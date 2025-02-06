from collections import defaultdict

import cv2
import numpy

from aicenter import img
from aicenter.log import get_module_logger
from aicenter.net import load_model, Result
from aicenter.sam import TrackingSAM

logger = get_module_logger(__name__)

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class AiCenter:
    def __init__(self, model=None, server=None, camera=None):
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.model_path = model

        # prepare neural network for detection
        self.net = load_model(model, CONF_THRESH, NMS_THRESH)

        # setup SAM2 for segmentation
        self.sam = TrackingSAM()

    def get_frame(self):
        try:
            data = self.video.get(self.key)
            image = numpy.frombuffer(data, numpy.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except TypeError as err:
            logger.error('Unable to grab frame')
            return
        else:
            return frame


    def process_frame(self, frame):
        if frame is not None:
            # Object detection
            height, width = frame.shape[:2]
            outputs = self.net.predict(frame)
            results = self.net.process_results(width, height, outputs)
            # Prompt segmentation with objects
            if results:
                for label, objects in results.items():
                    if label == 'loop' and objects and self.sam.predictor:
                        # Only use the highest-scoring loop as the prompt
                        loop = objects[0]
                        xyxy = [loop.x, loop.y, loop.x + loop.w, loop.y + loop.h]
                        input_boxes = numpy.atleast_2d(numpy.array(xyxy))
                        norm = numpy.array([width, height, width, height])
                        self.sam.track_input_boxes(frame, input_boxes, norm)
            # Segmentation
            if self.sam.init:
                masks, scores = self.sam.predict(frame)
                mask_results = self.sam.process_results(masks, scores, 'loop')
                if not results:
                    results = defaultdict(list)
                results['loop'].extend(mask_results)
                # Keep list sorted by score
                results['loop'] = sorted(results['loop'], key=lambda result: result.score, reverse=True)
            # Image processing fallback
            if not results:
                results = img.process_frame(frame)
            return results
