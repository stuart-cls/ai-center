import threading
import time
import warnings

import numpy
import redis

warnings.filterwarnings("ignore")

from enum import IntEnum

from devioc import models, log
import gepics

from . import AiCenter

logger = log.get_module_logger('aicenter')

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class EnableType(IntEnum):
    DISABLED, ENABLED = range(2)


class StatusType(IntEnum):
    VALID, INVALID = range(2)


# Create your models here. Modify the example below as appropriate
class AiCenterModel(models.Model):
    # Loop bounding box
    x = models.Integer('x', default=0, desc='X')
    y = models.Integer('y', default=0, desc='Y')
    w = models.Integer('w', default=0, desc='Width')
    h = models.Integer('h', default=0, desc='Height')
    score = models.Float('score', default=0.0, desc='Reliability')
    label = models.String('label', default='', desc='Object Type')
    status = models.Enum('status', choices=StatusType, desc="Status")
    enable = models.Enum('enable', choices=EnableType, default=1, desc="Enable/Disable")

    # Many-object centers
    objects_x = models.Array('objects:x', type=int, desc="Objects X")
    objects_y = models.Array('objects:y', type=int, desc="Objects Y")
    # objects_type = models.Array('objects:type', type=int, desc="Objects Type")
    objects_score = models.Array('objects:score', type=float, desc="Objects Score")
    objects_valid = models.Integer('objects:valid', default=0, desc="Valid objects")


class AiCenterApp(AiCenter):
    def __init__(self, device, model=None, server=None, camera=None):
        super().__init__(model=model, server=server, camera=camera)
        logger.info(f'device={device!r}, model={model!r}, server={server!r}, camera={camera!r}')
        self.running = False
        self.enabled = True
        self.ioc = AiCenterModel(device, callbacks=self)

        self.start_monitor()

    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0)
        while self.running:

            if not self.enabled:
                time.sleep(0.1)
                continue

            frame = self.get_frame()
            results = self.process_frame(frame)

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

                xs, ys, scores = [], [], []
                for label, res_list in results.items():
                    if label == 'loop':
                        continue
                    xs += [result.x + int(result.w / 2) for result in res_list]
                    ys += [result.y + int(result.h / 2) for result in res_list]
                    scores += [result.score for result in res_list]
                if xs:
                    self.ioc.objects_x.put(numpy.array(xs))
                    self.ioc.objects_y.put(numpy.array(ys))
                    self.ioc.objects_score.put(numpy.array(scores))
                    self.ioc.objects_valid.put(len(xs))
                else:
                    self.ioc.objects_valid.put(0)
            else:
                self.ioc.status.put(StatusType.INVALID)
                self.ioc.score.put(numpy.random.uniform(0, 0.01))
            time.sleep(0.001)

    def do_enable(self, pv, value, ioc):
        self.enabled = (value == EnableType.ENABLED)

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
