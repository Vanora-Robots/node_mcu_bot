from webcam import Webcam
import cv2
import sys
from datetime import datetime
import qr_tracker
import argparse as ap
from PyQt4 import QtCore, QtGui, uic

#form_class = uic.loadUiType("simple.ui")[0]
from dashboard_gui import MyWindowClass
from cnn_face_detector import Cnn_face_detector


QUADRILATERAL_POINTS = 4
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
GLYPH_PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1]

nodemcu_ip = "192.168.43.82:8080"
android_ip ="192.168.86.30:8080"


def nothing():
	print "blah"




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")

    args = vars(parser.parse_args())
    app = QtGui.QApplication(sys.argv)
    form_class = uic.loadUiType("../simple.ui")[0]
    w = MyWindowClass(None ,True, nodemcu_ip , android_ip )
    w.setWindowTitle('Robot AutoDocking')
    points = 1

    cv2.namedWindow('control')
    cv2.createTrackbar('thresh','control',5,255,nothing)
    cv2.createTrackbar('G','control',5,255,nothing)
    cv2.createTrackbar('B','control',5,255,nothing)

        # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'control',0,1,nothing)
    
    if args["videoFile"]:
        source = args["videoFile"]
    else:
        source = int(args["deviceID"])
    print("Device ID = /dev/video" + str(source))   
    webcam = Webcam(source , debug=True , root_url= android_ip)
    face_detector = Cnn_face_detector(debug=True , model ='mmod_human_face_detector.dat' )
    #webcam.start()
    #webcam.stop()
    while True:
        thresh = 100
        max_thresh = 255        

        
        thresh = cv2.getTrackbarPos('thresh','control')
        g = cv2.getTrackbarPos('G','control')
        b = cv2.getTrackbarPos('B','control')
        s = cv2.getTrackbarPos(switch,'control')

        if (w.get_running_status()):
            if (not (webcam.get_source() == 'webcam')):
                webcam.set_source('webcam')
            webcam.update_frame()
            retval,image = webcam.get_current_frame()
            if retval:

                w.update_frame_input(image)
                try:
                    drawing = webcam.thresh_callback(thresh , image)
                    w.update_frame_output(drawing)
                except:
                    print "failed to make the drawing from thresh callback function"
            else:
                print "Falied to retrive image"

        Phone_status , android_ip , port = w.get_Phone_status()

        if Phone_status:
            if webcam.get_source()=='ipcam':
                webcam.set_url(android_ip,port)
                webcam.set_source('ipcam')

            webcam.update_frame()
            retval,image = webcam.get_current_frame()
            if retval:

                w.update_frame_input(image)
                try:
                    drawing = webcam.thresh_callback(thresh , image)
                    w.update_frame_output(drawing)
                except:
                    print "failed to make the drawing from thresh callback function"


        if(w.get_tracker_status()):
            #print(webcam.get_points())
            if (not webcam.get_points()):
                print "Select points to be tracked"
                points = webcam.select_points()
                print("object tracker started-------------------------------------------------------------------")
            else :
                print webcam.get_points()
                retval,image = webcam.get_current_frame()
                if retval:
                    print "Cannot capture frame device | CODE TERMINATING :("
                    drawing = webcam.object_tracker(image)
                    w.update_frame_output(drawing)
        elif(not w.get_tracker_status()):
            webcam.points = None
        if(w.get_calibration_status()):
            webcam._update_frame()
            retval,image = webcam.get_current_frame() 
            if (retval):
                print "Calibration process initiated"
                ret = webcam.calibrate(1,8,6)
                w.update_frame_input(image)
                w.update_frame_output(ret)
            else :
                print("blah")
            
        if(w.get_test_calibration_status()):
            if (True):
                print "Test"
                image,test_result = webcam.checker_cube()
                print(image , test_result)
                if (image and test_result):
                    w.update_frame_input(image)
                    w.update_frame_output(test_result)
                else :
                    print("blah")

        if (w.get_cnn_face_detector_status()):
            if (True):
                print "Cnn face detector activated"
                webcam.update_frame()
                retval , image = webcam.get_current_frame()
                if retval:

                    w.update_frame_input(image)
                    try:
                      count,points = face_detector.detect_face(image)
                      print "Number of faces detected:",count
                    except:
                        print "failed to make the drawing from thresh callback function"
                else:
                    print "Falied to retrive image"

                    
        w.show()
                
        key = cv2.waitKey(10)%256
            #if (key):
            #   print("Key pressed:" + str(key))
            
        if key == 27:
            break
    
 
    
