from Common import Geometry


# calculate the number of cars crossing the lines
def counting(trk, start, end, cnt, index, results):
    flag = 0
    if trk.is_counted:
        return trk, cnt, results, flag
    cur_center = trk.cur_center
    pre_center = trk.pre_center
    cls = trk.cls
    if pre_center == []:
        return trk, cnt, results, flag
    if Geometry.is_segment_cross(start, end, pre_center, cur_center):
        flag = 1
        cnt[index] += 1
        if not trk.is_crossed_first_line:
            trk.crossed_line[0] = index
            trk.is_crossed_first_line = True
        elif index != trk.crossed_line[0]:
            trk.crossed_line[1] = index
            trk.is_crossed_second_line = True
            print("-->", cls-1, trk.crossed_line[0], trk.crossed_line[1])
            results[cls - 1][trk.crossed_line[0]][trk.crossed_line[1]] += 1
            trk.is_counted = True
    return trk, cnt, results, flag




















