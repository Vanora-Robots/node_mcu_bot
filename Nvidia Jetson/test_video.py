#!/usr/bin/env python
# Software License Agreement (BSD License)

import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import argparse
import sys

cap = cv2.VideoCapture("nvcamerasrc fpsRange='30.0 30.0' ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! nvvidconv flip-method=2 ! 'video/x-raw, format=(string)I420' ! videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink")

if not cap.isOpened():
    print "Camera did not open\n"
    sys.exit()

def traffic_light():
    pub = rospy.Publisher('stop_move', String, queue_size=10)
    rospy.init_node('traffic_light', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    stop_move = 'OFF'

    while not rospy.is_shutdown():
        _, frame = cap.read()

        #red
        lower_red = np.array([60, 60, 220], dtype=np.uint8)
        upper_red = np.array([110, 100, 255], dtype=np.uint8)
        
        #green
        lower_green = np.array([160, 250, 0], dtype=np.uint8)
        upper_green = np.array([255, 255, 120], dtype=np.uint8)
      
        mask_red = cv2.inRange(frame, lower_red, upper_red)
        res_red = cv2.bitwise_and(frame,frame, mask= mask_red)

        mask_green = cv2.inRange(frame, lower_green, upper_green)
        res_green = cv2.bitwise_and(frame,frame, mask= mask_green)

        kernel = np.ones((5,5),np.uint8)
        
        closing_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        blur_red = cv2.GaussianBlur(closing_red,(15,15),0)

        closing_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        blur_green = cv2.GaussianBlur(closing_green,(15,15),0)

        circles_red = cv2.HoughCircles(blur_red,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

        circles_green = cv2.HoughCircles(blur_green,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

        if circles_red is not None:
            # convert the (x, y) coordinates and radius of the circles to integers

            circles_red = np.round(circles_red[0, :]).astype("int")
                    
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles_red:
                if r > 10 and r < 30:
                    cv2.circle(frame, (x, y), r, (255, 0, 0), 2, 8, 0)
                    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    cv2.putText(frame,'Red Light-STOP!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 155)
                    cv2.imshow('gray',frame)
                    cv2.waitKey(3)
                    stop_move = 'stop1'

        if circles_green is not None:
            # convert the (x, y) coordinates and radius of the circles to integers

            circles_green = np.round(circles_green[0, :]).astype("int")
                    
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles_green:
                if r > 10 and r < 30:
                    cv2.circle(frame, (x, y), r, (255, 0, 0), 2, 8, 0)
                    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    cv2.putText(frame,'Green Light-GO!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 155)
                    cv2.imshow('gray',frame)
                    cv2.waitKey(3)
                    stop_move = 'move_1'

        cv2.imshow('frame',frame)
        cv2.waitKey(3)


##        cv2.imshow("Image window", cv_image)
        
        
        rospy.loginfo(stop_move)
        pub.publish(stop_move)
        rate.sleep()

if __name__ == '__main__':
    
    try:
        traffic_light()
    except rospy.ROSInterruptException:
        pass
