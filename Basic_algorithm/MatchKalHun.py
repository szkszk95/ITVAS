from Common import Geometry
from numpy import dot
from scipy.linalg import inv, block_diag
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np


# assign detections to trackers by linear assignment and Hungarian algorithm
def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = Geometry.box_iou(trk, det)

    # solve the maximizing the sum of IOU assignment problem using the Hungarian algorithm
    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if t not in matched_idx[:, 0]:
            unmatched_trackers.append(t)
    for d, det in enumerate(detections):
        if d not in matched_idx[:, 1]:
            unmatched_detections.append(d)
    matches = []
    for m in matched_idx:
        if IOU_mat[m[0], m[1]] < iou_thrd:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# class for Kalman Filter-based tracker
class Tracker():
    def __init__(self):

        # initialize
        self.id = 0  # tracker's id
        self.cur_box = []  # current bounding box
        self.pre_box = []  # previous bounding box
        self.cls = -1  # classification label

        self.is_crossed_first_line = False  # whether cross the first line
        self.is_crossed_second_line = False  # whether cross the second line
        self.crossed_line = [-1, -1]  # line index for two crossed lines
        self.is_counted = False  # whether be counted

        self.hits = 0  # number of detection matches
        self.no_losses = 0  # number of unmatched tracks
        self.x_state = []  # state
        self.dt = 1.  # time interval

        # process matrix (assuming constant velocity model)
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # measurement matrix (assuming only measure the coordinates)
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # state covariance
        self.L = 10.0
        self.P = np.diag(self.L * np.ones(8))

        # process covariance
        self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # measurement covariance
        self.R_scaler = 1.
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    # calculate the center of bounding box
    def get_center(self):
        if self.cur_box == []:
            self.cur_center = []
        else:
            # self.cur_center = [(self.cur_box[0] + self.cur_box[2]) / 2, (self.cur_box[1] + self.cur_box[3]) / 2]
            self.cur_center = [(self.cur_box[1] + self.cur_box[3]) / 2, (self.cur_box[0] + self.cur_box[2]) / 2]
        if self.pre_box == []:
            self.pre_center = []
        else:
            # self.pre_center = [(self.pre_box[0] + self.pre_box[2]) / 2, (self.pre_box[1] + self.pre_box[3]) / 2]
            self.pre_center = [(self.pre_box[1] + self.pre_box[3]) / 2, (self.pre_box[0] + self.pre_box[2]) / 2]

    # Kalman Filter for bounding box measurement

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    # predict and update
    def kalman_filter(self, z):
        x = self.x_state
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))
        y = z - dot(self.H, x)
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int)

    # only predict
    def predict_only(self):
        x = self.x_state
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)
