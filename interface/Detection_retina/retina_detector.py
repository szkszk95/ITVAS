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
                 model_path="models/vehicle_retinanet_94.pt",
                 class_path="models/clsVehicle.csv",
                 use_gpu=True):


        self.retinanet = torch.load(model_path)
        self.classes = self.load_classes(class_path)
        if use_gpu:
            self.retinanet = self.retinanet.cuda()
        self.retinanet.eval()

    def resizer(self, image, min_side=608, max_side=1024):
        import time
        t1 = time.time()
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

        return torch.from_numpy(new_image)

    def load_classes(self, path):
        csv_reader = csv.reader(open(path, 'r', newline=''), delimiter=',')
        result = {}

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
            img = skimage.color.gray2rgb(img)

        return self.resizer(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    def detect(self, img):
        import time
        result = []
        # unnormalize = UnNormalizer()

        def draw_caption(image, box, caption):
            b = np.array(box).astype(int)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        t1 = time.time()
        data = self.load_image(img)
        t2 = time.time()

        with torch.no_grad():
            scores, classification, transformed_anchors = self.retinanet(data.cuda().float())
            print("--->", t2-t1, time.time()-t2)
            idxs = np.where(scores > 0.5)

            img = np.array(255 * data[0, :, :, :]).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                label_name = self.classes[int(classification[idxs[0][j]])]
                result += [float(scores[idxs[0][j]]), x1, y1, x2, y2, label_name]
                print([float(scores[idxs[0][j]]), x1, y1, x2, y2, label_name])

                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow('img', img)
            cv2.waitKey(1)
        return []


if __name__ == '__main__':
    #
    # imgs = os.listdir("/data/00_share/image-3.18/")
    # for im in imgs:
    #     if im[-3:] == 'jpg':
    #         print(im)
    #         detect(cv2.imread("/data/00_share/image-3.18/"+im))

    # f = '/data/00_share/南通北200米监控/南通北200监控-16-17.mp4'
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
