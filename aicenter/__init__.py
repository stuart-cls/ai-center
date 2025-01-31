from collections import namedtuple, defaultdict

import cv2
import numpy

from aicenter import utils
from aicenter.net import load_model
from aicenter.sam import TrackingSAM

try:
    from devioc import log
except ImportError:
    import logging

    logger = logging.getLogger('aicenter')
else:
    logger = log.get_module_logger('aicenter')

# Result Type
Result = namedtuple('Result', 'type x y w h score')

CONF_THRESH, NMS_THRESH = 0.125, 0.25


class AiCenter:
    def __init__(self, model=None, server=None, camera=None):
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.model_path = model

        # prepare neural network for detection
        self.net = load_model(model, CONF_THRESH)
        self.net.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.layers = self.net.net.getLayerNames()
        self.output_layers = self.net.net.getUnconnectedOutLayersNames()
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

    def process_results(self, width, height, outputs):
        class_ids, confidences, bboxes = [], [], []
        for output in outputs:
            for bb, conf, cid in self.net.parse_output(output, width=width, height=height):
                bboxes.append(bb)
                confidences.append(conf)
                class_ids.append(cid)

        if bboxes:
            results = defaultdict(list)
            indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
            nms_boxes = [(bboxes[i], confidences[i], class_ids[i]) for i in indices]
            for bbox, score, class_id in nms_boxes:
                x, y, w, h = bbox
                label = self.net.names[class_id]
                logger.debug(f'{label} found at: {x} {y} [{w} {h}], prob={score}')
                results[label].append(Result(label, x, y, w, h, score))
            for label, llist in results.items():
                results[label] = sorted(llist, key=lambda result: result.score, reverse=True)
            return results

    @staticmethod
    def process_features(frame):
        """
        Process frame using traditional image processing techniques to detect loop
        :param frame: Frame to process
        :return: True if loop found
        """
        info = utils.find_loop(frame)
        if 'loop-x' in info:
            logger.debug(
                f'Loop found at: {info["loop-x"]} {info["loop-y"]} [{info["loop-width"]} {info["loop-height"]}]'
            )
            return {
                'loop': [
                    Result('img-loop', info['x']-25, info['y']-25, 50, 50, 0.25 + numpy.random.uniform(0, 0.001))
                ]
            }

    def process_frame(self, frame):
        if frame is not None:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (self.net.size, self.net.size), swapRB=True, crop=False)
            self.net.net.setInput(blob)
            outputs = self.net.net.forward(self.output_layers)
            results = self.process_results(width, height, outputs)
            if not results:
                # attempt regular image processing
                results = self.process_features(frame)
            return results
