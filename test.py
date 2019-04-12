from UI.proc import *
from interface.Detection_retina.retina_detector import RetinaNet


def proc(video_path, lines, frame_gap=1, debug=False):
    cap = cv2.VideoCapture(video_path)
    frame_num = -1
    results = np.zeros((6, len(lines), len(lines))).astype(np.uint16)       # count results
    cnt = np.zeros(len(lines)).astype(np.uint16)                            # number of vehicles crossing each line
    retinanet = RetinaNet("/home/szk/PycharmProjects/pytorch-retinanet/saved/resnet50_vehicle_39.pt")

    while True:
        ret, frame = cap.read()
        frame_num += 1

        # interval frame processing
        if frame_num % frame_gap != 0:
            continue
        print(frame_num)

        # begin to detect, track and count
        if ret:
            frame, cnt, results = pipeline(frame, frame_num, lines, cnt, results, debug, retinanet)

            # show_frame = frame
            print("--"*40)
            # print('results', results)
            if debug:
                height, width = frame.shape[:2]

                cv2.imshow("video", frame)
                cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    proc(
        video_path="/data/00_share/4天视频/01   227省道、东港路（4天）/4.11/227省道、东港路西北角_227省道、东港路西北角_20180411070000.mp4",
        lines=[[484, 385, 1606, 832], [282, 490, 1139, 851]],
        frame_gap=1
    )
