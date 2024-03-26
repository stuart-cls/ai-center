import os
import threading
import time
import warnings

import cv2
import numpy
import redis

warnings.filterwarnings("ignore")

from enum import IntEnum
from collections import namedtuple, defaultdict

from devioc import models, log
import gepics

from . import utils
logger = log.get_module_logger('aicenter')

# Result Type
Result = namedtuple('Result', 'type x y w h score')

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class StatusType(IntEnum):
    VALID, INVALID = range(2)


# Create your models here. Modify the example below as appropriate
class AiCenter(models.Model):
    # Loop bounding box
    x = models.Integer('x', default=0, desc='X')
    y = models.Integer('y', default=0, desc='Y')
    w = models.Integer('w', default=0, desc='Width')
    h = models.Integer('h', default=0, desc='Height')
    score = models.Float('score', default=0.0, desc='Reliability')
    label = models.String('label', default='', desc='Object Type')
    status = models.Enum('status', choices=StatusType, desc="Status")
    # Many-object centers
    objects_x = models.Array('objects:x', type=int, desc="Objects X")
    objects_y = models.Array('objects:y', type=int, desc="Objects Y")
    # objects_type = models.Array('objects:type', type=int, desc="Objects Type")
    objects_score = models.Array('objects:score', type=float, desc="Objects Score")
    objects_valid = models.Integer('objects:valid', default=0, desc="Valid objects")


class AiCenterApp(object):
    def __init__(self, device, model=None, server=None, camera=None):
        logger.info(f'device={device!r}, model={model!r}, server={server!r}, camera={camera!r}')
        self.running = False
        self.ioc = AiCenter(device, callbacks=self)
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.model_path = model

        # prepare neural network for detection
        with open(os.path.join(model, 'yolov3.names'), 'r', encoding='utf-8') as fobj:
            names = [line.strip() for line in fobj.readlines()]

        self.darknet = {
            'weights': os.path.join(model, 'yolov3.weights'),
            'config': os.path.join(model, 'yolov3.cfg'),
            'names': names,
        }

        self.net = cv2.dnn.readNetFromDarknet(self.darknet['config'], self.darknet['weights'])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.layers = self.net.getLayerNames()
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        # self.output_layers = [self.layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.start_monitor()

    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

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
            for detection in output:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    cx, cy, w, h = (detection[0:4] * numpy.array([width, height, width, height])).astype(int)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    bboxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        if bboxes:
            results = defaultdict(list)
            indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
            nms_boxes = [(bboxes[i], confidences[i], class_ids[i]) for i in indices]
            for bbox, score, class_id in nms_boxes:
                x, y, w, h = bbox
                label = self.darknet['names'][class_id]
                logger.debug(f'{label} found at: {x} {y} [{w} {h}], prob={score}')
                results[label].append(Result(label, x, y, w, h, score))
            for label, llist in results.items():
                results[label] = sorted(llist, key=lambda result: result.score, reverse=True)
            self.ioc.status.put(StatusType.VALID)
            return results
        else:
            self.ioc.status.put(StatusType.INVALID)
            self.ioc.score.put(0.0)

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
            return {'loop': Result('loop', info['loop-x'], info['loop-y'], info['loop-width'], info['loop-height'], 0.5)}

    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0)
        while self.running:
            frame = self.get_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                results = self.process_results(width, height, outputs)
                if not results:
                    # attempt regular image processing
                    results = self.process_features(frame)

                if results:
                    if 'loop' in results:
                        # Only return highest-scoring loop
                        result = results['loop'][0]
                        self.ioc.x.put(result.x)
                        self.ioc.y.put(result.y)
                        self.ioc.w.put(result.w)
                        self.ioc.h.put(result.h)
                        self.ioc.label.put(result.type)
                        self.ioc.score.put(result.score)
                        self.ioc.status.put(StatusType.VALID)
                    xs = [], ys = [], scores = []
                    for label, reslist in results.values():
                        if label == 'loop':
                            continue
                        xs += [result.x + int(result.w / 2) for result in reslist]
                        ys += [result.y + int(result.h / 2) for result in reslist]
                        scores += [result.score for result in reslist]
                    if xs:
                        self.ioc.objects_x.put(numpy.array(xs))
                        self.ioc.objects_y.put(numpy.array(ys))
                        self.ioc.objects_score.put(numpy.array(scores))
                        self.ioc.objects_valid.put(len(xs))
                    else:
                        self.ioc.objects_valid.put(0)
                else:
                    self.ioc.status.put(StatusType.INVALID)
                    self.ioc.score.put(0.0)
            else:
                self.ioc.status.put(StatusType.INVALID)
                self.ioc.score.put(0.0)
            time.sleep(0.001)

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
