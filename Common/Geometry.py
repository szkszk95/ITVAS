import cv2
import numpy as np


# test whether two line segments intersect
def cross_product(A, B):
    return A[0] * B[1] - A[1] * B[0]


def rapid_rejection_test(A1, A2, B1, B2):
    return (min(A1[0], A2[0]) <= max(B1[0], B2[0])) and (min(B1[0], B2[0]) <= max(A1[0], A2[0])) and \
           (min(A1[1], A2[1]) <= max(B1[1], B2[1])) and (min(B1[1], B2[1]) <= max(A1[1], A2[1]))

def is_segment_cross(A1, A2, B1, B2):
    # print("RAPID:", rapid_rejection_test(A1, A2, B1, B2))
    if not rapid_rejection_test(A1, A2, B1, B2):
        return False
    AB = [A2[0] - A1[0], A2[1] - A1[1]]
    AC = [B1[0] - A1[0], B1[1] - A1[1]]
    AD = [B2[0] - A1[0], B2[1] - A1[1]]
    CA = [A1[0] - B1[0], A1[1] - B1[1]]
    CB = [A2[0] - B1[0], A2[1] - B1[1]]
    CD = [B2[0] - B1[0], B2[1] - B1[1]]
    if cross_product(AB, AC) * cross_product(AB, AD) <= 0 and cross_product(CD, CA) * cross_product(CD, CB) <= 0:
        return True
    return False

# calculate iou of two bounding box
def box_iou(a, b):
    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])
    if float((s_a + s_b - s_intsec)) == 0.0 or float(s_intsec) == 0.0:
        return 0.0
    return float(s_intsec) / float((s_a + s_b - s_intsec))

def get_cls(cls):
    if cls == 1:
        return 'minibus'
    elif cls == 2:
        return 'bus'
    elif cls == 3:
        return 'smalltruck'
    elif cls == 4:
        return 'mediumtruck'
    elif cls == 5:
        return 'largetruck'
    elif cls == 6:
        return 'trailer'
    elif cls == -1:
        return "no_tracked"
    #return "vehicle"
    #
    # if cls == 1:
    #     return 'minibus_smalltruck'
    # elif cls == 2:
    #     return 'bus'
    # elif cls == 3:
    #     return 'medium_large_truck_trailer'

# draw bounding boxes if debug == True
def draw_box_label(frame, bbox_cv2, id, cls):
    box_color = (int(id * 20) % 255, int(id * 80) % 255, int(id * 40) % 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
    cv2.rectangle(frame, (left - 2, top - 10), (right + 2, top), box_color, -1, 1)
    text_x = str(id) + ':' + get_cls(cls)
    cv2.putText(frame, text_x, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
    return frame

# draw lines if debug == True
def draw_lines(frame, lines, flag):
    for j in range(len(lines)):
        if flag[j] >= 1:
            # print("bian")
            cv2.line(frame, (lines[j][0], lines[j][1]), (lines[j][2], lines[j][3]), (255, 0, 255), 5)
            cv2.putText(frame, str(j + 1), (lines[j][0], lines[j][1]), 1, 3, (128, 0, 0), 3, 8)
        else:
            cv2.line(frame, (lines[j][0], lines[j][1]), (lines[j][2], lines[j][3]), (0, 255, 0), 5)
            cv2.putText(frame, str(j + 1), (lines[j][0], lines[j][1]), 1, 3, (128, 0, 0), 3, 8)
    return frame






