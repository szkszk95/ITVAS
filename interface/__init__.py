from __future__ import absolute_import


from .Detection_retina.retina_detector import RetinaNet
from .Detection.fasterrcnn_detector import FasterRCNN


__factory = {
    'fasterRCNN': FasterRCNN,
    'retinaNet': RetinaNet
}


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError('please use a legal model')
    return __factory[name](*args, **kwargs)
