from UI.MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from UI.proc import proc
import cv2
import time
import os
# from interface.Detection import detector


class ui(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ui, self).__init__()
        self.setupUi(self)
        self.x1, self.x2 = 0, 0
        self.y1, self.y2 = 0, 0
        self.lines = []
        self.files = []
        self.font_font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        self.font_size = 4
        self.line_gap = 15
        self.is_drawing = False
        self.is_clicked = False
        self.show_frame = None
        self.save_file = "."
        self.video_capture = cv2.VideoCapture()

        self.pushButton.clicked.connect(self.open_video)
        self.pushButton_2.clicked.connect(self.add_video)
        self.pushButton_3.clicked.connect(self.draw_line)
        self.pushButton_4.clicked.connect(self.save_config)
        self.pushButton_5.clicked.connect(self.delete_line)
        self.pushButton_6.clicked.connect(self.run)
        self.pushButton_7.clicked.connect(self.choose_model)
        # self.show()

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "打开文件",
                                                  ".",
                                                  "mp4 Files(*.mp4);;ALL Files(*)")
        if filename == '':
                self.show_messages(['警告', '没有选择文件！'])
        else:
            print(filename)

            self.video_capture.open(filename)
            ret, frame = self.video_capture.read()
            self.show_frame = frame
            height, width = frame.shape[:2]

            if frame.ndim == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            image = image.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            self.label.setPixmap(QPixmap.fromImage(image))

    def add_video(self):
        filenames, _ = QFileDialog.getOpenFileNames(self,
                                                    "打开文件",
                                                    ".",
                                                    "mp4 Files(*.mp4);;ALL Files(*)")
        print(filenames)
        self.textBrowser.setText("选择视频：")
        for fn in filenames:
            if fn not in self.files:
                self.files.append(fn)
        for file in self.files:
            self.textBrowser.append(file)

    def choose_model(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                    "打开文件",
                                                    ".",
                                                    "pth Files(*.pth);;ALL Files(*)")
        if filename == '':
            self.show_messages(['警告', '没有选择文件！'])

        else:
            print(filename)
            self.model_path = filename

    def save_config(self):
        if len(self.lines) == 0:
            self.show_messages(['Warning', "Please draw lines first!"])
            return
        elif len(self.files) == 0:
            self.show_messages(['Warning', "Please add videos first!"])
            return
        filename = QFileDialog.getExistingDirectory(self,
                                                    "Save",
                                                    "/data/yxy/")
        if filename == "":
            print("...")
        else:
            # fp = open(filename, "w")
            # for fn in self.files:
            #     fp.write(fn+"\n")
            # fp.write(str(len(self.lines))+"\n")
            # for line in self.lines:
            #     for i in range(4):
            #         fp.write(str(line[i])+"\t")
            #     fp.write("\n")
            # fp.close()
            self.save_file = filename

    def draw_line(self):
        if self.video_capture is None:
            self.show_messages(['警告!', "请先添加视频!"])
        self.is_drawing = True
        print(self.x(), self.label.x(), self.label.pos().y(), self.verticalLayout_3.Down)
        print("start draw line")

    def delete_line(self):
        if len(self.lines) > 0:
            self.lines.pop()
        else:
            self.show_messages(['警告!', '没有线条可以删除!'])
        temp = self.show_frame.copy()
        height, width = temp.shape[:2]
        for i in range(len(self.lines)):
            x1, y1, x2, y2 = self.lines[i]
            cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.putText(temp, str(i+1), (x1, y1 - self.line_gap),
                        self.font_font, self.font_size, (0, 0, 255), 2)

        if temp.ndim == 3:
            rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
        image = image.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap.fromImage(image))

    def set_pos(self, height, width):
        x1 = int(self.x1 / self.label.width() * width)
        x2 = int(self.x2 / self.label.width() * width)
        y1 = int(self.y1 / self.label.height() * height)
        y2 = int(self.y2 / self.label.height() * height)
        return x1, y1, x2, y2

    def paintEvent(self, event):
        event.parent = self.label
        if self.is_drawing and self.is_clicked:
            temp = self.show_frame.copy()
            height, width = temp.shape[:2]
            temp = cv2.resize(temp, (width, height))

            # draw old lines
            for i in range(len(self.lines)):
                x1, y1, x2, y2 = self.lines[i]
                cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(temp, str(i+1), (x1, y1 - self.line_gap),
                            self.font_font, self.font_size, (0, 0, 255), 2)
            # draw new line
            x1, y1, x2, y2 = self.set_pos(height, width)
            cv2.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.putText(temp, str(len(self.lines)+1), (x1, y1 - self.line_gap),
                        self.font_font, self.font_size, (255, 0, 0), 2)

            if temp.ndim == 3:
                rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            image = image.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(QPixmap.fromImage(image))

    def mousePressEvent(self, event):
        if self.is_drawing:
            self.is_clicked = True
            self.x1 = event.pos().x()
            self.y1 = event.pos().y()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.x2 = event.pos().x()
            self.y2 = event.pos().y()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.is_clicked = False
            temp = self.show_frame.copy()
            height, width = temp.shape[:2]
            for i in range(len(self.lines)):
                x1, y1, x2, y2 = self.lines[i]
                cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(temp, str(i+1), (x1, y1 - self.line_gap),
                            self.font_font, self.font_size, (0, 0, 255), 2)
            x1, y1, x2, y2 = self.set_pos(height, width)
            cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.putText(temp, str(len(self.lines)+1), (x1, y1 - self.line_gap),
                        self.font_font, self.font_size, (0, 0, 255), 2)

            if temp.ndim == 3:
                rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            image = image.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            self.label.setPixmap(QPixmap.fromImage(image))
            self.lines.append([x1, y1, x2, y2])

    def show_messages(self, msg):
        QMessageBox.information(self, msg[0], msg[1], QMessageBox.Yes)

    def write_flow(self, video, result):
        file = os.path.join(self.save_file,
                            video.split("/")[-1].split(".")[0]+".txt")
        fp = open(file, "w")
        print(result.shape)
        fp.write("\t轿车\t公交车\t小型货车\t中型货车\t大型货车\t拖挂车\n")
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                if i == j:
                    continue
                fp.write(str(i+1)+"-"+str(j+1)+"\t")
                for category in range(result.shape[0]):
                    fp.write(str(result[category, i, j])+"\t")
                fp.write("\n")
        return

    def run(self):
        self.textBrowser.append("开始处理视频:")
        # load tf model

        for video in self.files:
            self.textBrowser.append(video+"处理中...")
            t1 = time.time()
            print(self.checkshow.isChecked())
            result = proc(label=self.label,
                          video=video,
                          lines=self.lines,
                          model_path=self.model_path,
                          gap=self.spinBox.value(),
                          debug=self.checkshow.isChecked())

            t2 = time.time()
            self.write_flow(video, result)
            self.textBrowser.append("{:d} minutes".format(int((t2-t1)/60)))
            self.textBrowser.append("处理完成...")
