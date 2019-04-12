import cv2
import numpy as np
import os
import time
import sys

from Basic_algorithm import DetectionRCNN
from Basic_algorithm import MatchKalHun
from Common import Geometry, Count
import interface

from collections import deque
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

max_age = 10  # number of consecutive unmatched detection before a track is deleted
min_hits = 2  # number of consecutive matches needed to establish a track
tracker_list = []  # list for trackers
track_id_list = deque(range(0, 999999))  # list for track ID


# pipeline of detecting, tracking, and counting
def pipeline(frame, frame_num, lines, cnt, results, if_show, model):
    """

    :param frame:
    :param frame_num:
    :param lines:
    :param cnt:
    :param results:
    :param if_show:
    :param model:
    :return:
    """
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    original_frame = frame

    # detect
    time_detect = time.time()
    detections = DetectionRCNN.start_detect(frame, frame_num, model)

    # track
    time_track = time.time()
    z_box = []
    x_box = []
    clss = []
    for i in range(len(detections)):
        z_box.append(detections[i][3:7])
        clss.append(detections[i][7])
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.cur_box + [trk.id])
    matched, unmatched_dets, unmatched_trks = MatchKalHun.assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)

    # Deal with matched detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            pre_box = tmp_trk.cur_box
            tmp_trk.pre_box = pre_box
            tmp_trk.cur_box = xx
            # tmp_trk.cls = clss[det_idx]
            if tmp_trk.cls != 5 and tmp_trk.cls != 6:
                tmp_trk.cls = clss[det_idx]
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = MatchKalHun.Tracker()
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            # tmp_trk.pre_box = z
            tmp_trk.cur_box = xx
            tmp_trk.cls = clss[idx]
            tmp_trk.id = track_id_list.popleft()

            # new id add into the track list
            tracker_list.append(tmp_trk)

    # Deal with unmatched tracks
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            pre_box = tmp_trk.cur_box
            tmp_trk.pre_box = pre_box
            tmp_trk.cur_box = xx

    # count and show
    good_tracker_list = []
    flag = np.zeros(len(lines))
    for trk in tracker_list:
        if (trk.hits >= min_hits) and (trk.no_losses <= max_age):
            frame = Geometry.draw_box_label(frame, trk.cur_box, trk.id, trk.cls)
            trk.get_center()
            for j in range(len(lines)):
                trk, cnt, results, flag_temp = Count.counting(trk, lines[j][0:2], lines[j][2:4], cnt, j, results)
                flag[j] += flag_temp
            good_tracker_list.append(trk)
    if if_show:
        frame = Geometry.draw_lines(frame, lines, flag)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    print("Detection\t{:2f}\ttraciking\t{:2f}".format(
        time_track - time_detect,
        time.time() - time_track
    ))

    return frame, cnt, results


def proc(label, video, lines, model_path, gap=1, if_show=False):
    """
    Process every Frame
    :param label: THE QT SHOW Label
    :param video: Video Path
    :param lines: THE LINES
    :param model_path: Path to the model
    :param gap: Process one each <gap> frames
    :param if_show: if SHOW result in the app
    :return:
    """
    cap = cv2.VideoCapture(video)
    frame_num = -1
    results = np.zeros((6, len(lines), len(lines))).astype(np.uint16)  # count results
    cnt = np.zeros(len(lines)).astype(np.uint16)  # number of vehicles crossing each line

    if 'retina' in model_path:
        net = interface.create('retinaNet', model_path=model_path)
    elif 'yolo' in model_path:
        print("yolo not ready yet!")
        return
    elif 'rcnn' in model_path:
        net = interface.create('fasterRCNN', model_path=model_path)
    else:
        print("not this model!")
        return

    while True:
        ret, frame = cap.read()
        frame_num += 1

        # interval frame processing
        if frame_num % gap != 0:
            continue
        print(frame_num)

        # begin to detect, track and count
        if ret:
            print("--" * 40)

            # Detection. Tracking. Counting Pipeline
            frame, cnt, results = pipeline(frame, frame_num, lines, cnt, results, if_show, net)

            if if_show:
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                image = image.scaled(label.width(), label.height(), Qt.KeepAspectRatio)

                label.setPixmap(QPixmap.fromImage(image))
                QCoreApplication.processEvents()
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return results
