from UI.ui_window import ui
import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

# sys.path.append("/data/szk/PycharmProjects/IVAN_v2")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = ui()
    mainWindow.show()
    sys.exit(app.exec_())
