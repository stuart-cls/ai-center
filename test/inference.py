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

CONF_THRESH, NMS_THRESH = 0.25, 0.25


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
            print('Unable to grab frame')
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

    @staticmethod
    def process_features(frame):
        """
        Process frame using traditional image processing techniques to detect loop
        :param frame: Frame to process
        :return: True if loop found
        """
        info = find_loop(frame)
        if "loop-x" in info:
            print(f'Loop found at: {info["loop-x"]} {info["loop-y"]} [{info["loop-width"]} {info["loop-height"]}]')
            return Result('loop', info['loop-x'], info['loop-y'], info['loop-width'], info['loop-height'], 1.0)

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
                if not res:
                    res = self.process_features(frame)

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


def find_loop(orig, offset=10, scale=0.5, orientation='left'):
    raw = cv2.flip(orig, 1) if orientation != 'left' else orig
    y_max, x_max = orig.shape[:2]
    frame = cv2.resize(raw, (0, 0), fx=scale, fy=scale)

    clean = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 11, 11)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    edges = cv2.bitwise_not(cv2.dilate(thresh, None, 10))
    avg, stdev = cv2.meanStdDev(gray)

    edges[:offset, :] = 0
    edges[-offset:, :] = 0
    edges[:, -offset:] = 0
    height, width = edges.shape
    tip_x, tip_y = width // 2, height // 2

    info = {
        'mean': avg,
        'std': stdev,
        'signal': avg / stdev,
        'found': 0,                 # 0 = nothing found, 1 = tip found, 2 = ellipse fitted.
        'center-x': tip_x / scale,
        'center-y': tip_y / scale,
        'score': 0.0,
    }

    if edges.max() > 10:
        info['found'] = 1
        prof = numpy.argwhere(edges.T > 128)
        cols, indices = numpy.unique(prof[:, 0], return_index=True)
        data = numpy.split(prof[:, 1], indices[1:])
        profiles = numpy.zeros((len(cols), 5), int)
        for i, arr in enumerate(data):
            mini, maxi = arr.min(), arr.max()
            profiles[i, :] = (cols[i], mini, maxi, maxi - mini, (maxi + mini) // 2)
            cv2.line(edges, (cols[i], mini), (cols[i], maxi), (128, 0, 255), 1)

        search_width = width / 8
        idx = 3
        valid = (
            (numpy.abs(profiles[:, idx] - profiles[:, idx].mean()) < 2 * profiles[:, idx].std())
            & (profiles[:, idx] < 0.8 * height)
        )
        if valid.sum() > 5:
            profiles = profiles[valid]

        tip_x = profiles[:, 0].max()
        tip_y = profiles[profiles[:, 0].argmax(), 4]

        info['x'] = tip_x / scale
        info['y'] = tip_y / scale
        valid = (profiles[:, 0] >= tip_x - search_width)

        vertices = numpy.concatenate((
            profiles[:, (0, 1)][valid],
            profiles[:, (0, 2)][valid][::-1]
        )).astype(int)
        sizes = profiles[:, 3][valid]

        if len(vertices) > 5:
            center, size, angle = cv2.fitEllipse(vertices)
            c_x, c_y = center
            s_x, s_y = size
            if abs(c_y - tip_y) > height // 2 or s_x >= width or s_y >= height:
                center, size, angle = cv2.minAreaRect(vertices)
            info['found'] = 2
            info['ellipse'] = (
                tuple([int(x / scale) for x in center]),
                tuple([int(x / scale) for x in size]),
                angle,
            )

            ellipse_x, ellipse_y = info['ellipse'][0]
            ellipse_w, ellipse_h = max(info['ellipse'][1]), min(info['ellipse'][1])

            info['loop-x'] = int(ellipse_x)
            info['loop-y'] = int(ellipse_y)
            info['loop-width'] = ellipse_w
            info['loop-height'] = ellipse_h
            info['loop-angle'] = angle

            info['loop-start'] = ellipse_x + info['loop-width']/2
            info['loop-end'] = ellipse_x - info['loop-width']/2
            info['score'] = 100*(1 - abs(info['loop-start'] - info['x'])/info['loop-width'])

        info['sizes'] = (sizes / scale).astype(int)
        info['points'] = [(int(x / scale), int(y / scale)) for x, y in vertices]

    else:
        info['x'] = 0
        info['y'] = info['center-x']

    if orientation == 'right':
        for k in ['loop-x', 'loop-start', 'loop-end', 'x']:
            if k in info:
                info[k] = x_max - info[k]
        if 'points' in info:
            info['points'] = [(x_max - x, y) for x, y in info['points']]

    return info


if __name__ == '__main__':
    app = AiCenterApp(model="/cmcf_apps/ai-centering/model", server="IOC1608-304.clsi.ca", camera="0030180F06E5")
    app.run()
