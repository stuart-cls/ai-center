import os.path
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Iterator

import cv2
import numpy
import yaml

from aicenter.log import get_module_logger

logger = get_module_logger(__name__)

# Result Type
Result = namedtuple('Result', 'type x y w h score')

class Net:
    size = None
    net = None
    names = None

    def __init__(self, model_path, conf_threshold: float, nms_threshold: float):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.load_model()
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def load_model(self):
        """Model-specific loading"""
        raise NotImplementedError

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        """Model-specific output handling"""
        raise NotImplementedError

    def predict(self, image: numpy.ndarray) -> numpy.ndarray:
        blob = cv2.dnn.blobFromImage(image, 0.00392, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.output_layers)

    def process_results(self, width, height, outputs):
        class_ids, confidences, bboxes = [], [], []
        for output in outputs:
            for bb, conf, cid in self.parse_output(output, width=width, height=height):
                bboxes.append(bb)
                confidences.append(conf)
                class_ids.append(cid)

        if bboxes:
            results = defaultdict(list)
            indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.conf_threshold, self.nms_threshold).flatten()
            nms_boxes = [(bboxes[i], confidences[i], class_ids[i]) for i in indices]
            for bbox, score, class_id in nms_boxes:
                x, y, w, h = bbox
                label = self.names[class_id]
                logger.debug(f'{label} found at: {x} {y} [{w} {h}], prob={score}')
                results[label].append(Result(label, x, y, w, h, score))
            for label, llist in results.items():
                results[label] = sorted(llist, key=lambda result: result.score, reverse=True)
            return results

class DarkNet(Net):
    size = 416

    def load_model(self):
        model_path = str(self.model_path)
        with open(os.path.join(model_path, 'yolov3.names'), 'r', encoding='utf-8') as fobj:
            self.names = [line.strip() for line in fobj.readlines()]
        self.net = cv2.dnn.readNetFromDarknet(
            os.path.join(model_path, 'yolov3.cfg'),
            os.path.join(model_path, 'yolov3.weights'),
        )

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        for detection in output:
            scores = detection[5:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_threshold:
                cx, cy, w, h = (detection[0:4] * numpy.array([width, height, width, height])).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                yield [x, y, int(w), int(h)], float(confidence), int(class_id)


class ONNXNet(Net):
    size = 640

    def load_model(self):
        self.model_path = Path(self.model_path)
        with open(next(self.model_path.glob('*.yaml')), 'r') as fobj:
            data = yaml.safe_load(fobj)
        self.names = list(data['names'].values())
        onnx = str(next(self.model_path.glob('*.onnx')))
        self.net = cv2.dnn.readNetFromONNX(onnx)

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        for i in range(output.shape[-1]):
            detection = output[0, ..., i]
            scores = detection[4:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_threshold:
                scale = numpy.array([width, height, width, height]) / self.size
                cx, cy, w, h = (detection[0:4] * scale).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                yield [x, y, int(w), int(h)], float(confidence), int(class_id)


def load_model(model_path: str or Path, conf_threshold: float, nms_threshold: float) -> Net:
    for n in [DarkNet, ONNXNet]:
        try:
            net = n(model_path, conf_threshold, nms_threshold)
        except OSError:
            continue
        else:
            return net
    raise ValueError('No such model')
