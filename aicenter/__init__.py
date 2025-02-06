import cv2
import numpy

from aicenter import utils
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
            outputs = self.net.predict(frame)
            results = self.net.process_results(width, height, outputs)
            if not results:
                # attempt regular image processing
                results = self.process_features(frame)
            return results
