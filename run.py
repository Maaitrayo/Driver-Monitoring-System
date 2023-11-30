import sys
import cv2
import numpy as np

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from detect import detect_face

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.run_flag:
            ret, frame = cap.read()
            frame = detect_face(frame)
            if ret:
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self.run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5 video')
        self.disply_width = 2000
        self.display_height = 900
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.text_label = QLabel("Sometext")

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(self.text_label)
        self.setLayout(self.vbox)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        self.thread.start() 

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    
    @pyqtSlot(np.ndarray)
    def update_image(self, img):
        qt_img = self.convert_cv_qt(img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())