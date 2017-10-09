from webcam import Webcam
import cv2
from datetime import datetime
import numpy as np



with np.load('calibration/webcam_calibration_ouput.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

focal_length = 640
center = (320 , 240)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
dist_coeffs = np.zeros((4,1))

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])

#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (359, 391),     # Nose tip
                            (399, 561),     # Chin
                            (337, 297),     # Left eye left corner
                            (513, 301),     # Right eye right corne
                            (345, 465),     # Left Mouth corner
                            (453, 469)      # Right mouth corner
                        ], dtype="double")
 

 # 2d points in image plane.

webcam = Webcam()
#webcam.start()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
 
while True:
     
    # get image from webcam
    webcam.update_frame()
    image = webcam.get_current_frame()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((3*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:3].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = []
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    
 
    # save image to file, if pattern found
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,3), None)
    
    if  True:
        print("Found chess board-------------------------------------------------")
        objpoints.append(objp)
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        #imgpoints.append(corners2)
        #img = cv2.drawChessboardCorners(image, (5,3), corners2,ret)
        # display image
        #cv2.imshow('grid', img)
        #print(img)
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        #np.savez("calibration/calib", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        
        #print(objp, corners2)
        rvecs, tvecs, inliers = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,flags=cv2.CV_ITERATIVE)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        print("Complete")
        
        
    
        cv2.imshow('img',img)
    k = cv2.waitKey(10)%256
    print("input",k)
    if k == 113:
        print("ending the program , thank you ")
        exit()