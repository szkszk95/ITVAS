import numpy as np
import sys
import cv2
import csv

import torch
import skimage.io
import skimage.transform
import skimage.color
import skimage

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import interface.Detection_retina.model as model

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class RetinaNet:
    def __init__(self,
                 model_path,
                 class_path="models/clsVehicle.csv",
                 use_gpu=True):

        self.retinanet = torch.load(model_path)
        self.classes = self.load_classes(class_path)
        if use_gpu:
            self.retinanet = self.retinanet.cuda()
        self.retinanet.eval()

    def resizer(self, image, min_side=608, max_side=1024):
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round((cols * scale))), int(round(rows * scale))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return torch.from_numpy(new_image).permute(2, 0, 1).unsqueeze(0), pad_h, pad_w

    def load_classes(self, path):
        result = {}
        csv_reader = csv.reader(open(path, 'r', newline=''), delimiter=',')
        for line, row in enumerate(csv_reader):
            line += 1
            class_name, class_id = row

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[int(class_id)] = class_name
        return result

    def load_image(self, image):
        img = image[:, :, ::-1]

        if len(img.shape) == 2:
            # img = skimage.color.gray2rgb(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return self.resizer(img.astype(np.float32) / 255.0)

    def detect(self, img):
        result = []
        ori_H, ori_W = img.shape[:2]

        shaped, pad_h, pad_w = self.load_image(img)
        H, W = shaped.shape[2:]

        with torch.no_grad():
            scores, classification, transformed_anchors = self.retinanet(shaped.cuda().float())
            idxs = np.where(scores > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0]*ori_W/(W-pad_w))
                y1 = int(bbox[1]*ori_H/(H-pad_h))
                x2 = int(bbox[2]*ori_W/(W-pad_w))
                y2 = int(bbox[3]*ori_H/(H-pad_h))
                # label_name = self.classes[int(classification[idxs[0][j]])]
                result.append([float(scores[idxs[0][j]])*100, x1, y1, x2, y2, int(classification[idxs[0][j]])+1])
        print(result)
        return result


if __name__ == '__main__':
    f = "/data/00_share/04  绸都大道、舜湖西路（4天）/4.11/绸都大道、舜湖西路南侧_绸都大道、舜湖西路南侧_20180411074813.mp4"

    retinanet = RetinaNet()
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
            result = retinanet.detect(im_in)
            print(result)
