#!/usr/bin/env python3

import os
import threading
import time
import warnings

import cv2
import numpy
import redis

warnings.filterwarnings("ignore")

from enum import Enum
from collections import namedtuple

# Result Type
Result = namedtuple('Result', 'type x y w h score')

CONF_THRESH, NMS_THRESH = 0.5, 0.5


class StatusType(Enum):
    VALID, INVALID = range(2)


class AiCenterApp(object):
    def __init__(self, model=None, server=None, camera=None):
        print(f'model={model!r}, server={server!r}, camera={camera!r}')
        self.running = False
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
            indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
            scores = [confidences[index] for index in indices]
            index = indices[numpy.argmax(scores)]
            x, y, w, h = bboxes[index]
            score = confidences[index]
            label = self.darknet['names'][class_ids[index]]
            print(f'{label} found at: {x} {y} [{w} {h}], prob={score}')
            return Result(label, x, y, w, h, score)

    def run(self, scale=0.5):
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0)
        while self.running:
            raw_frame = self.get_frame()
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if frame is not None:
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                res = self.process_results(width, height, outputs)
                if res is not None:
                    cv2.rectangle(frame, (res.x, res.y), (res.x+res.w, res.y+res.h), (255, 0, 0), 1)
                    cv2.putText(frame, f'{res.type}:{res.score:0.2f}', (res.x, res.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False


if __name__ == '__main__':
    app = AiCenterApp(model="/cmcf_apps/ai-center/model", server="IOC1608-304.clsi.ca", camera="0030180F06E5")
    app.run()
