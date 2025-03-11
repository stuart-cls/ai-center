import os.path
from pathlib import Path
from typing import Iterator

import cv2
import numpy
import yaml


class Net:
    size = None
    net = None
    names = None

    def __init__(self, model_path, conf_thres):
        self.model_path = model_path
        self.conf_thres = conf_thres

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        raise NotImplementedError


class DarkNet(Net):
    size = 416

    def __init__(self, model_path, conf_thres):
        super().__init__(model_path, conf_thres)
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

            if confidence > self.conf_thres:
                cx, cy, w, h = (detection[0:4] * numpy.array([width, height, width, height])).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                yield [x, y, int(w), int(h)], float(confidence), int(class_id)


class ONNXNet(Net):
    size = 640

    def __init__(self, model_path, conf_thres):
        super().__init__(model_path, conf_thres)
        self.model_path = Path(model_path)
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

            if confidence > self.conf_thres:
                scale = numpy.array([width, height, width, height]) / self.size
                cx, cy, w, h = (detection[0:4] * scale).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                yield [x, y, int(w), int(h)], float(confidence), int(class_id)


def load_model(model_path: str or Path, conf_thres: float) -> Net:
    for n in [DarkNet, ONNXNet]:
        try:
            net = n(model_path, conf_thres)
        except OSError:
            continue
        else:
            return net
    raise ValueError('No such model')
