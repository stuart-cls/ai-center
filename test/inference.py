#!/usr/bin/env python3

import warnings

import cv2
import redis

from aicenter import AiCenter

warnings.filterwarnings("ignore")

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class AiCenterApp(AiCenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'model={self.model_path!r}, server={self.server!r}, camera={self.key!r}')
        self.running = False

    def run(self, scale=0.5):
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0)
        while self.running:
            raw_frame = self.get_frame()
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            results = self.process_frame(frame)

            if results is not None:
                for res in results:
                    cv2.rectangle(frame, (res.x, res.y), (res.x+res.w, res.y+res.h), (255, 0, 0), 1)
                    cv2.putText(frame, f'{res.type}:{res.score:0.2f}', (res.x, res.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    app = AiCenterApp(model="/cmcf_apps/ai-centering/model", server="IOC1608-304.clsi.ca", camera="0030180F06E5")
    app.run()
