from __future__ import absolute_import

import sys

from .Detection_retina.retina_detector import RetinaNet
from .Detection.fasterrcnn_detector import FasterRCNN
#sys.path.insert(0,'/home/ztc/Projects/ITVAS/IVAN_pytorch/interface/Detection/lib')

__factory = {
    'fasterRCNN':FasterRCNN,
    'retinaNet':RetinaNet
}

def create(name,*args,**kwargs):
    if name not in __factory:
        raise KeyError('please use a legal model')
    return __factory[name](*args,**kwargs)