from PyQt4 import QtCore, QtGui, uic, Qwt5
from PyQt4.Qt import *
from PyQt4.QtCore import *
from PyQt4.Qwt5 import *
from PyQt4.Qwt5.qplt import *
import sys
import cv2
import numpy as np
import threading
import time
import Queue

from connect_Bot import *

form_class = uic.loadUiType("../simple.ui")[0]
running = False
object_tracker = False
calibration = False


class sliderdemo(QWidget):
    def __init__(self, parent=None):
        super(sliderdemo, self).__init__(parent)
        layout = QVBoxLayout()
        self.l1 = QLabel("Hello")
        self.l1.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.l1)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(10)
        self.sl.setMaximum(30)
        self.sl.setValue(20)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(5)
        layout.addWidget(self.sl)
        self.sl.valueChanged.connect(self.valuechange)
        self.setLayout(layout)
        self.setWindowTitle("SpinBox demo")

    def valuechange(self):
        size = self.sl.value()
        self.l1.setFont(QFont("Arial", size))


class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None, debug=False, nodemcu_ip="192.168.43.82:8080", android_ip="192.168.43.1:8080"):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.startButton.clicked.connect(self.start_clicked)
        self.trackerButton.clicked.connect(self.tracker_clicked)
        self.calibrateButton.clicked.connect(self.calibrate_clicked)
        self.testcalibrateButton.clicked.connect(self.testcalibrate_clicked)
        self.faceDetector.clicked.connect(self.cnn_face_detector)
        self.targetButton.clicked.connect(self.target_tracker)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.tracker_width = self.tracker.frameSize().width()
        self.tracker_height = self.tracker.frameSize().height()

        self.ImgWidget = OwnImageWidget(self.ImgWidget)
        self.OutputWidget = OwnImageWidget(self.OutputWidget)
        self.tracker = OwnImageWidget(self.tracker)
        self.running = False
        self.object_tracker = False
        self.target_Tracker = False
        self.calibration = False
        self.test_calibration = False
        self.cnn_face_detector_status = False
        self.debug = debug

        self.ConnectBot.clicked.connect(self.start_connectBot)
        self.ConnectPhone.clicked.connect(self.start_connectPhone)
        self.nodemcu_ip = nodemcu_ip.split(":")[0]
        self.nodemcu_port = nodemcu_ip.split(":")[1]
        self.android_port = android_ip.split(":")[0]
        self.android_ip = android_ip.split(":")[1]
        self.NodeIP.setText(nodemcu_ip)
        self.PhoneIP.setText(android_ip)
        self.ConnectBot_status = False
        self.Phone_status = False

    def start_clicked(self):
        self.running = not self.running
        if self.running:
            self.startButton.setEnabled(True)
            self.startButton.setText('Stop')
            print("Video Stream on", self.running)
        else:
            self.startButton.setEnabled(True)
            self.startButton.setText('Start')
            print("Video Stream off", self.running)

    def get_running_status(self):
        return self.running

    def tracker_clicked(self):
        self.object_tracker = not self.object_tracker
        if self.object_tracker:
            self.trackerButton.setEnabled(True)
            self.trackerButton.setText('Tracker on')
            print("Object Tracker on", self.object_tracker)
        else:
            self.trackerButton.setEnabled(True)
            self.trackerButton.setText('Tracker off')
            print("Object Tracker Off", self.object_tracker)

    def get_tracker_status(self):
        return self.object_tracker

    def target_tracker(self):
        self.target_Tracker = not self.target_Tracker
        if self.target_Tracker:
            self.targetButton.setText("Target ON")
        if  not self.target_Tracker:
            self.targetButton.setText("Target OFF")
    def get_target_tracker(self):
        return self.target_Tracker

    def calibrate_clicked(self):
        self.calibration = not self.calibration
        if self.calibration:
            self.calibrateButton.setEnabled(True)
            self.calibrateButton.setText('Calibration on')
            print("Object Tracker on", self.calibration)
        else:
            self.calibrateButton.setEnabled(True)
            self.calibrateButton.setText('calibration off')
            print("Object Tracker Off", self.calibration)

    def get_calibration_status(self):
        return self.calibration

    def testcalibrate_clicked(self):
        self.test_calibration = not self.test_calibration
        if self.test_calibration:
            self.testcalibrateButton.setEnabled(True)
            self.testcalibrateButton.setText('Calibration on')
            print("Object Tracker on", self.test_calibration)
        else:
            self.testcalibrateButton.setEnabled(True)
            self.testcalibrateButton.setText('calibration off')
            print("Object Tracker Off", self.test_calibration)

    def get_test_calibration_status(self):
        return self.test_calibration

    def update_frame_input(self, input_img):

        img = input_img
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            print("updating the input frame")
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)

    def update_frame_output(self, output_img):

        img = output_img
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            print("updating the output frame")
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.OutputWidget.setImage(image)
    def update_tracker_output(self, output_img):

        img = output_img
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.tracker_width_width) / float(img_width)
        scale_h = float(self.tracker_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            print("updating the output frame")
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_Indexed8)
        self.tracker.setImage(image)

    def closeEvent(self, event):
        global running
        running = False

    def start_connectBot(self):
        input = self.NodeIP.toPlainText()
        input = str(input)
        print "IP address", input.split(":")[0]
        print "IP address", input.split(":")[1]
        #self.connectBot = Connect_Bot(input.split(":")[0], input.split(":")[1])
        self.ConnectBot_status = True
        self.nodemcu_ip = input.split(":")[0]
        self.nodemcu_port = input.split(":")[1]

    def get_Connect_bot_status(self):
        return self.ConnectBot_status, self.NodeIP.toPlainText().split(":")[0], self.NodeIP.toPlainText().split(":")[1]

    def start_connectPhone(self):
        self.Phone_status = not self.Phone_status
        input = self.PhoneIP.toPlainText()
        input = str(input)
        print "IP address", input.split(":")[0]
        print "IP address", input.split(":")[1]
        self.Phone_status = True
        self.android_ip = input.split(":")[0]
        self.android_port = input.split(":")[1]

    def get_Phone_status(self):
        return self.Phone_status, self.android_ip, self.android_port

    def cnn_face_detector(self):
        self.cnn_face_detector_status = not self.cnn_face_detector_status
        if (self.cnn_face_detector_status):
            self.faceDetector.setText("Face detector ON")
        else:
            self.faceDetector.setText("Face detector OFF")
        print ("Started CNN Face detector")

    def get_cnn_face_detector_status(self):
        return self.cnn_face_detector_status

