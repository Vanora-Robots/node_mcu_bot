from webcam import Webcam
from computer_vision import Computer_vison
from ipcam import Ipcam
import cv2
import sys ,math
import helper


import argparse as ap
from PyQt4 import QtCore, QtGui, uic
import numpy as np


from dashboard_gui import MyWindowClass
from cnn_face_detector import Cnn_face_detector

from connect_Bot import *


QUADRILATERAL_POINTS = 4
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
GLYPH_PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1]
connect_bot = 0





thresh = 100
max_thresh = 255


def nothing():
	print ""

def main(w , webcam, ip_cam , computer_vision, face_detector):
    thresh = cv2.getTrackbarPos('thresh', 'control')
    g = cv2.getTrackbarPos('G', 'control')
    b = cv2.getTrackbarPos('B', 'control')
    s = cv2.getTrackbarPos(switch, 'control')
    global connect_bot

    ####################################################################################
    ##USB WEB CAM
    if w.get_running_status():

        webcam.update_frame()
        retval, image = webcam.get_current_frame()
        if retval:

            w.update_frame_input(image)
            try:
                drawing = computer_vision.thresh_callback(thresh, image)
                w.update_frame_output(drawing)
            except:
                print "failed to make the drawing from thresh callback function"
        else:
            print "Falied to retrive image"
    #####################################################################################



    ######################################################################################
    ##IPCAM
    try:
        Phone_status, port, android_ip = w.get_Phone_status()

    except:
        Phone_status = None
        android_ip = None
        port = None

    if Phone_status:
        image = ip_cam.get_IP_image()

        if image:
            print "Image "
            w.update_frame_input(image)
            try:
                drawing = computer_vision.thresh_callback(thresh, image)
                w.update_frame_output(drawing)
            except:
                print "failed to make the drawing+++ from thresh callback function"

    ###################################################################################


    ###################################################################################
    ##NODEMCU##################################
    try:
        status, ip, port = w.get_Connect_bot_status()
        if status:
            connect_bot = Connect_Bot(ip, port , mode= "Normal")
            #print "Connected with the bot"
        else:
            a=0
            #connect_bot.update()

    except:
        print "Bot not connected"
    ###################################################################################



    ###################################################################################
    ##OBJECT TRACKER DLIB CORRELATION TRACKER##########################################
    if (w.get_tracker_status()):
        # print(webcam.get_points())
        if(webcam.get_status()):
            webcam.update_frame()
            retval, image = webcam.get_current_frame()
        if (not computer_vision.get_points()):
            print "Select points to be tracked"
            points = computer_vision.select_points(image)
            print("object tracker started-------------------------------------------------------------------")
        elif (not connect_bot):
            status , ip , port=w.get_Connect_bot_status()
            connect_bot = Connect_Bot(ip , port ,mode="Manual")
        else:

            retval, image = webcam.get_current_frame()
            if retval:
                drawing, tracker , pt1 , pt2 = computer_vision.object_tracker(image)
                print "Points pt1:pt2,",pt1,":" , pt2

                ctrd = (np.array(pt1)+np.array(pt2))/2
                area = (np.array(pt1)-np.array(pt2))
                area = area[0]*area[1]
                print "Area:ctrd ,",area,":", ctrd
                if (ctrd)[0]<320:
                    connect_bot.move_bot('R')
                elif(ctrd)[0]>320:
                    connect_bot.move_bot('L')
                elif area < 12000 :
                    connect_bot.move_bot('F')
                elif area > 12000:
                    connect_bot.move_bot('B')

                w.update_frame_output(drawing)
                w.update_tracker_output(tracker)
    #######################################################################################


    #######################################################################################
    ##STATIC TRACKER#######################################################################
    if(w.get_target_tracker()):
        webcam.update_frame()
        retval, image = webcam.get_current_frame()
        try:
            w.update_frame_input(image)
        except:
            print "Failed to retrieve image from camera"
        area, output = computer_vision.get_area(image)
        w.update_frame_output(output)
        print area
        try:

            marker = computer_vision.find_marker(output)
            print marker
            KNOWN_DISTANCE = 24.0

            # initialize the known object width, which in this case, the piece of
            # paper is 11 inches wide
            KNOWN_WIDTH = 11.0
            focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
            inches = computer_vision.distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
            print inches
            #print math.sqrt(((9)*area)/600)

        except:
            print "Failed to get area"
    ############################################################################################



    ############################################################################################
    ##CALIBRATE USING CHECKERBOARD##############################################################
    if (w.get_calibration_status()):
        webcam._update_frame()
        retval, image = webcam.get_current_frame()
        if (retval):
            print "Calibration process initiated"
            ret = computer_vision.calibrate(1, 8, 6)
            w.update_frame_input(image)
            w.update_frame_output(ret)
        else:
            print("blah")
    ##############################################################################################

    ##############################################################################################
    ##TEST CALIBRATION############################################################################
    if (w.get_test_calibration_status()):
        if (True):
            print "Test"
            image, test_result = computer_vision.checker_cube()
            print(image, test_result)
            if (image and test_result):
                w.update_frame_input(image)
                w.update_frame_output(test_result)
            else:
                print("blah")


    ##############################################################################################
    ##DLIB CNN FACE DETECTOR######################################################################
    if (w.get_cnn_face_detector_status()):
        if (True):
            print "Cnn face detector activated"
            webcam.update_frame()
            retval, image = webcam.get_current_frame()
            if retval:

                w.update_frame_input(image)
                try:
                    count, points = face_detector.detect_face(image)
                    print "Number of faces detected:", count
                except:
                    print "failed to run face_detector.detect_face"
            else:
                print "Falied to retrive image"
    #################################################################################################

    #################################################################################################
    key = cv2.waitKey(10) % 256



    if key == 27:
        return 0
    else:
        return w, webcam, ip_cam, computer_vision, face_detector

    ##################################################################################################
    ##END OF MAIN


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")

    args = vars(parser.parse_args())

    ############################################
    nodemcu_ip = "192.168.43.82:8080"
    android_ip = "192.168.86.30:8080"
    ############################################


    points = 1

    ########################################################################
    webcam = Webcam(debug=False, name='webcam')
    try:
        list_of_video_devices = webcam.get_source_list(path='/dev/video*')
        webcam.set_source(0)
    except:
        print "Webcam not detected"
    #########################################################################


    #########################################################################
    print ("Initiated Ip cam as ", android_ip)
    ip_cam = Ipcam(root_url=android_ip, debug=True)
    #########################################################################





    ######################################################
    try:
        computer_vision = Computer_vison(debug=True)
        print ("Initiated Computer vision module")
    except:
        print ("Please debug CV module")
    ######################################################

    ################################################################################################
    face_detector = Cnn_face_detector(debug=False, model='darknet/mmod_human_face_detector.dat')
    ################################################################################################

    ################################################################################################



    ######################################################
    app = QtGui.QApplication(sys.argv)
    print ("UI initiating ............... ")
    try:
        w = MyWindowClass(None, debug=False, nodemcu_ip=nodemcu_ip, android_ip=android_ip)
        print "Window Class loaded"
        w.setWindowTitle('Robot AutoDocking')
        print "...................Finished UI init"
    except:
        print "Failed to initiate the UI"
    ######################################################

    ###########################################################
    cv2.namedWindow('control')
    cv2.createTrackbar('thresh', 'control', 5, 255, nothing)
    cv2.createTrackbar('G', 'control', 5, 255, nothing)
    cv2.createTrackbar('B', 'control', 5, 255, nothing)

        # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'control', 0, 1, nothing)

    thresh = cv2.getTrackbarPos('thresh', 'control')
    g = cv2.getTrackbarPos('G', 'control')
    b = cv2.getTrackbarPos('B', 'control')
    s = cv2.getTrackbarPos(switch, 'control')

    print ("Created the CV2")
    ###########################################################




    while True:
        try:
            main(w, webcam, ip_cam, computer_vision, face_detector)
            #t = threading.Thread(target=main , args= (w, webcam, ip_cam, computer_vision, face_detector))
            #t.deamon = True
            #t.start()
        except:
            print "Failed main"

        try:
            w.show()
        except:
            print "Failed UI"





