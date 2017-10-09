from webcam import Webcam
import cv2
from datetime import datetime
import numpy as np
import os , time




webcam = Webcam(0)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
 
while True:
     
    # get image from webcam
    #os.system('clear')
    webcam.update_frame()
    retval , image = webcam.get_current_frame()
    if retval:
        webcam.display_img(image,"raw_image", delay=10)
        gray = webcam.to_gray(image)
        webcam.display_img(gray,"grayscale",delay=10)
        thresh_img = webcam.extract_bright(gray,False)
        webcam.display_img(thresh_img,"threshhold",delay=10)
        led_img, regions , cntrs = webcam.find_leds(thresh_img)
        if led_img:
            webcam.display_img(led_img,"ledIMage", delay=10)
            for cnt in cntrs:
                webcam.display_img(cnt, str(i), delay=10)
            centers = webcam.leds_positions(regions)
            # webcam.showStats(centers)
            print "Total number of Leds found : %d !" % (len(centers))
            print "###"
            print "Led positions :"
            for c in centers:
                print "x : %d; y : %d" % (c[0], c[1])
            print "###"
        else:
            print ("No Led found")
        i = 0

    

    #time.sleep(0.1)

    k = cv2.waitKey(100)%256
    #print("input",k)
    if k == 113:
        print("ending the program , thank you ")
        exit()