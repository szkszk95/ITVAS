from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
sys.path.insert(0,osp.abspath(osp.dirname(__file__)+osp.sep+'..'+osp.sep+'..'))

#import Detection._init_paths
import interface.Detection._init_paths
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

xrange = range  # Python 3



os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
#
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='interface/Detection/cfgs/res101.yml', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    args = parser.parse_args()
    return args


class FasterRCNN(object):

    def __init__(self,
                 model_path='../../models/epoch_12_10.24.pth',
                 use_gpu=True,
                 gpu_num=0
                 ):

        self.args=parse_args()

        if self.args.cfg_file is not None:
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)

        self.use_gpu=use_gpu
        if self.use_gpu:
            cfg.USE_GPU_NMS = 1
        else:
            cfg.USE_GPU_NMS = 0

        np.random.seed(cfg.RNG_SEED)

        self.pascal_classes = np.asarray(['__background__',  # always index 0
                                     'minibus',
                                     'largetruck',
                                     'smalltruck',
                                     'mediumtruck',
                                     'bus',
                                     'trailer'])

        self.fasterRCNN=resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.args.class_agnostic)
        self.fasterRCNN.create_architecture()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        checkpoint = torch.load(model_path)

        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        if use_gpu:
            self.fasterRCNN.cuda()
        self.fasterRCNN.eval()

        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        if self.use_gpu:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        self.im_data = Variable(self.im_data, volatile=True)
        self.im_info = Variable(self.im_info, volatile=True)
        self.num_boxes = Variable(self.num_boxes, volatile=True)
        self.gt_boxes = Variable(self.gt_boxes, volatile=True)

        if self.use_gpu:
            cfg.CUDA = True


    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)



    def detect(self, image, threshold=0.5, max_bbox=20):
        thresh = 0.05
        vis = True
        result = []

        # rgb -> bgr
        im = image[:, :, ::-1]

        blobs, im_scales = self._get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.data.resize_(1, 1, 5).zero_()
        self.num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.args.class_agnostic:

                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

                    box_deltas = box_deltas.view(1, -1, 4)
                else:

                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        for j in xrange(1, len(self.pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    dets = cls_dets.cpu().numpy()
                    for i in range(np.minimum(max_bbox, dets.shape[0])):
                        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
                        score = dets[i, -1]
                        if score > threshold:
                            ymin, xmin, ymax, xmax = bbox
                            result.append(score)
                            result.append(int(xmin))
                            result.append(int(ymin))
                            result.append(int(xmax))
                            result.append(int(ymax))
                            result.append(self.pascal_classes[j])

        return result


if __name__ == '__main__':
    #
    # imgs = os.listdir("/data/00_share/image-3.18/")
    # for im in imgs:
    #     if im[-3:] == 'jpg':
    #         print(im)
    #         detect(cv2.imread("/data/00_share/image-3.18/"+im))

    # f = '/data/00_share/南通北200米监控/南通北200监控-16-17.mp4'
    f = "/data/00_share/04  绸都大道、舜湖西路（4天）/4.11/绸都大道、舜湖西路南侧_绸都大道、舜湖西路南侧_20180411074813.mp4"

    net=FasterRCNN()
    cap = cv2.VideoCapture(f)

    ret = True
    curFrame = -1
    while ret:
        ret, image = cap.read()
        if not ret:
            break
        curFrame += 1

        im_in = np.array(image)
        if curFrame % 2 == 0:
            result = net.detect(im_in)
            print(result)