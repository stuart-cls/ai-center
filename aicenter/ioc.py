import cv2
import os
import numpy
import time
import redis
import warnings
import threading


warnings.filterwarnings("ignore")
from multiprocessing import cpu_count
from operator import attrgetter
from collections import namedtuple, defaultdict
from darkflow.net.build import TFNet

from devioc import models, log
import gepics

logger = log.get_module_logger('aicenter')
SCALE = 4
MODEL_DIR = os.environ.get('AICENTER_MODEL', os.path.join(os.path.dirname(__file__), 'model'))

# Result Type
Result = namedtuple('Result', 'type x y w h score')

# Create your models here. Modify the example below as appropriate

class AiCenter(models.Model):
    loop_x = models.Integer('loop:x', default=0, desc='Loop X')
    loop_y = models.Integer('loop:y', default=0, desc='Loop Y')
    loop_w = models.Integer('loop:w', default=0, desc='Loop Width')
    loop_h = models.Integer('loop:h', default=0, desc='Loop Height')
    loop_score = models.Float('loop:score', default=0.0, desc='Loop Reliability')

    xtal_x = models.Integer('xtal:x', default=0, desc='Crystal X')
    xtal_y = models.Integer('xtal:y', default=0, desc='Crystal Y')
    xtal_w = models.Integer('xtal:w', default=0, desc='Crystal Width')
    xtal_h = models.Integer('xtal:h', default=0, desc='Crystal Height')
    xtal_score = models.Float('xtal:score', default=0.0, desc='Crystal Reliability')
    
# create your app here. Modify the following example as appropriate

class AicenterApp(object):
    def __init__(self, device_name, server=None, camera=None, scale=4):
        self.running = False
        self.scale = scale
        self.ioc = AiCenter(device_name, callbacks=self)
        self.key = f'{camera}:JPG'
        self.video = redis.Redis(host=server, port=6379, db=0)

        os.chdir(MODEL_DIR)
        self.tfnet = TFNet({
            'model': os.path.join(MODEL_DIR, 'tiny-yolo-1c.cfg'),
            'load': -1,
            'labels': os.path.join(MODEL_DIR, 'labels.txt'),
            'threshold': 0.3,
            'cpu': cpu_count()
        })
        self.start_monitor()
    
    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

    def get_frame(self):
        data = self.video.get(self.key)
        image = numpy.frombuffer(data, numpy.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return cv2.resize(frame, (0,0), fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_AREA)  
        
    def get_results(self, results):
        return [
            Result(
                result["label"],
                self.scale * int((result['topleft']['x'] + result['bottomright']['x'])/2),
                self.scale * int((result['topleft']['y'] + result['bottomright']['y'])/2),
                self.scale *  abs(result['bottomright']['x'] - result['topleft']['x']),
                self.scale *  abs(result['bottomright']['y'] - result['topleft']['y']),
                round(result['confidence'],2),
            )
            for result in results
        ]
    
    def process_results(self, results):
        types = defaultdict(list)
        for result in results:
            types[result.type].append(result)
        
        ordered_types = {
            name: sorted(items, key=attrgetter('score'), reverse=True)
            for name, items in types.items()
        }
        
        if ordered_types.get('loop'):
            loop = ordered_types['loop'][0]
            self.ioc.loop_x.put(loop.x)
            self.ioc.loop_y.put(loop.y)
            self.ioc.loop_w.put(loop.w)
            self.ioc.loop_h.put(loop.h)
            self.ioc.loop_score.put(loop.score) 
            logger.debug(f"Loop found: {loop}")
            
        if ordered_types.get('xtal'):
            xtal = ordered_types['xtal'][0]
            self.ioc.xtal_x.put(xtal.x)
            self.ioc.xtal_y.put(xtal.y)
            self.ioc.xtal_w.put(xtal.w)
            self.ioc.xtal_h.put(xtal.h)
            self.ioc.xtal_score.put(xtal.score)
            logger.debug(f"Crystal found: {xtal}")
            
            
    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        while self.running:
            frame = self.get_frame()
            results = self.get_results(self.tfnet.return_predict(frame))
            self.process_results(results)
            time.sleep(0.001)
    
    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
        
