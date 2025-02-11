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
    def __init__(self, model=None, server=None, camera=None, conf_thresh=CONF_THRESH):
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.model_path = model

        # prepare neural network for detection
        self.net = load_model(model, conf_thresh, NMS_THRESH)

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
                self.sam.track_objects(frame, results, width, height)
            # Segmentation
            if self.sam.tracked_objects:
                outputs = self.sam.predict(frame)
                mask_results = self.sam.process_results(*outputs)
                if not results:
                    results = defaultdict(list)
                for label in mask_results.keys():
                    results[label].extend(mask_results[label])
                    # Keep list sorted by score
                    results[label] = sorted(results[label], key=lambda result: result.score, reverse=True)
            # Image processing fallback
            if not results:
                results = img.process_frame(frame)
            return results
