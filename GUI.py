import sys
from os import path

import cv2
import numpy as np
import requests
import json 

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QTabWidget, QWidget, QPushButton, QHBoxLayout,QVBoxLayout,QSlider,QLabel,QMessageBox,QListWidget,QTextEdit

addr = 'http://localhost:5000'
test_url = addr + '/post'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture("./video1.avi")

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            img = cv2.resize(data,(480,360))
            self.image_data.emit(img)
            
            _, img_encoded = cv2.imencode('.jpg', img)
        
            # response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
            # print(response.text)
         
class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (640, 480)
        self.setFixedSize(640,480)
    # def detect_faces(self, image: np.ndarray):
    #     # haarclassifiers work better in black and white
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     gray_image = cv2.equalizeHist(gray_image)

    #     faces = self.classifier.detectMultiScale(gray_image,
    #                                              scaleFactor=1.3,
    #                                              minNeighbors=4,
    #                                              flags=cv2.CASCADE_SCALE_IMAGE,
    #                                              minSize=self._min_size)

    #     return faces

    def image_data_slot(self, image_data):
        # faces = self.detect_faces(image_data)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image_data,
        #                   (x, y),
        #                   (x+w, y+h),
        #                   self._red,
        #                   self._width)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(640,480)

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget()
        
        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)
        
        self.list_widget = QListWidget()

        main_layout_1 = QVBoxLayout()

        layout = QtWidgets.QHBoxLayout()
        layout_left = QtWidgets.QVBoxLayout()
        layout_right = QtWidgets.QHBoxLayout()

        layout_left.addWidget(self.face_detection_widget)
        
        layout_right.addWidget(self.list_widget)

        self.run_button = QtWidgets.QPushButton('Start')
        layout_left.addWidget(self.run_button)
        
        self.run_button.clicked.connect(self.record_video.start_recording)


        layout.addLayout(layout_left)
        layout.addLayout(layout_right)


        tabs = QTabWidget()

        tab1 = QWidget()
        tab2 = QWidget()

        tab1.setLayout(layout)

        tab2.setLayout(layout)

        tabs.addTab(tab1,"Run mode")
        tabs.addTab(tab2,"Train mode")

        main_layout_1.addWidget(tabs)
        self.setLayout(main_layout_1)



def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))

    main()