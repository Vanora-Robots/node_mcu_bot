import cv2
from glyphfunctions import *
import numpy as np
import dlib

import get_points



class Computer_vison:
    def __init__(self , debug = False ):
        print "Initialized the Computer vison module"
        self.std = None
        self.debug = debug
        self.thread = None
        self.points = None
        self.tracker = None

    def to_gray(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grey_img

    def grey_histogram(self,img, nBins=64):
        h = np.zeros((300, 256, 3))

        bins = np.arange(256).reshape(256, 1)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([img], [ch], None, [256], [0, 255])
            cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
            hist = np.int32(np.around(hist_item))
            pts = np.column_stack((bins, hist))
            cv2.polylines(h, [pts], False, col)

        h = np.flipud(h)
        if (self.debug):
            cv2.imshow('colorhist', h)
            cv2.waitKey(0)

        return h

    def extract_bright(self, grey_img, histogram=False):

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_img)

        if 0:

            hist = self.grey_histogram(grey_img, nBins=64)
            [hminValue, hmaxValue, hminIdx, hmaxIdx] = cv.GetMinMaxHistValue(hist)
            margin = 0  # statistics to be calculated using hist data
        else:
            margin = 0.8

        thresh = int(maxVal * margin)

        ret, thresh_img = cv2.threshold(grey_img, maxVal - minVal, maxVal, cv2.THRESH_BINARY)

        return thresh_img

    def find_leds(self, thresh_img):

        try:
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            return None, None, None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        regions = []
        i = 0
        cntrs = []
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append((x, y, w, h))
            # self.display_img(self.current_frame[x:x+w,y:y:h],str(i)  , delay = 10)
            i = i + 1
            print(self.current_frame[x:x + w, y:y + h])
            cntrs.append(self.current_frame[x:x + w, y:y:h])

        print cntrs
        return thresh_img, regions, cntrs

    def leds_positions(self, regions):

        centers = []
        for x, y, width, height in regions:
            centers.append([x + (width / 2), y + (height / 2)])

        return centers

    def display_img(self, image, name, delay=1000):
        print ("size  , ", image.size)
        if (not image is None):
            cv2.imshow(name, image)
            cv2.waitKey(delay)

    def thresh_callback(self, thresh ,image ):

        if self.debug:
            print ("started thesh_callback")


        QUADRILATERAL_POINTS = 4
        SHAPE_RESIZE = 100.0
        BLACK_THRESHOLD = 100
        WHITE_THRESHOLD = 155
        GLYPH_PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1]

        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        output = cv2.GaussianBlur(output, (5, 5), 0)

        #output = cv2.Canny(output, thresh , thresh*thresh)
        output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        output = cv2.medianBlur(output, 3)

        #ret , output = cv2.threshold(output , 127 , 255 , 0)
        contours = []

        try:
            if(self.debug):
                cv2.imshow("output" , output)
                cv2.waitKey(10)
            contours , hierarchy = cv2.findContours(output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if(self.debug):
                print("failed to find contours")
            try:
                hierarchy = hierarchy[0]
            except:
                hierarchy = []
            if(self.debug):
                print "hierarchy", hierarchy
        except:
            return image

        #contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        if(self.debug):
            print ("contours =" , contours.count)
        if not hierarchy:
            print "contours not  found"
            return output
        else:
            print ("countours found")
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
                print approx
                if len(approx) == QUADRILATERAL_POINTS:
                    topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))
                    resized_shape = resize_image(topdown_quad, SHAPE_RESIZE)
                    if resized_shape[5, 5] > BLACK_THRESHOLD: continue
                glyph_found = False
                for i in range(4):
                    glyph_pattern = get_glyph_pattern(resized_shape, BLACK_THRESHOLD, WHITE_THRESHOLD)
                    if glyph_pattern == GLYPH_PATTERN:
                        glyph_found = True
                        print "glyph found -------------------------------------------------------"
                        break

                    resized_shape = rotate_image(resized_shape, 90)

                if glyph_found:
                    substitute_image = cv2.imread('1.jpg')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    break
            return output

    def select_points(self , image):
        self.points = get_points.run(image)
        self.tracker = dlib.correlation_tracker()
        points = self.points
        self.tracker.start_track(image, dlib.rectangle(*points[0]))
        return points

    def get_points(self):
        return self.points

    def object_tracker(self, image):
        points = self.points
        self.tracker.update(image)
        rect = self.tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))

        cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)


        height = np.size(image,0)
        width = np.size(image,1)

        tracker = np.zeros((height , width))
        cv2.rectangle(tracker, pt1, pt2, (255, 255, 255), 3)

        loc = (int(rect.left()), int(rect.top() - 20))
        txt = "Object tracked at [{}, {}]".format(pt1, pt2)
        if(self.debug):
            cv2.putText(image, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", image)
            cv2.imshow("tracker", tracker)
            print "Object tracked at [{}, {}] \r".format(pt1, pt2),
        return image ,tracker ,pt1 , pt2

    def calibrate(self, n_boards, board_w, board_h):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        img = self.current_frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if ret == True:
            print("Checker board found")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        return img

    def check_calibration(self):
        intrinsic = cv.Load("Intrinsics.xml")
        distortion = cv.Load("Distortion.xml")
        print " loaded all distortion parameters"
        image = cv.fromarray(self.current_frame)
        mapx = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1);
        mapy = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1);
        cv.InitUndistortMap(intrinsic, distortion, mapx, mapy)
        cv.NamedWindow("Undistort")
        print "all mapping completed"
        print "Now relax for some time"
        print "now get ready, camera is switching on"
        self._update_frame()
        image = self.current_frame
        map_x = np.array(mapx)
        map_y = n
        p.array(mapy)
        t = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        t = cv.fromarray(t)
        c = cv.WaitKey(33)
        print "everything is fine"
        return image, np.array(t)

    def checker_cube(self):

        with np.load('calibration/webcam_calibration_ouput.npz',  mmap_mode='r') as X:
            mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        img = self.current_frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        if ret == True:
            print("Eureka , checkerboard found")
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = self.draw(img, corners2, imgpts)
            cv2.imshow('img', img)
            cv2.waitKey(10)
            return self.current_image, img
        else:
            return 0, 0

    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def qr_tracker(image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 100, 200)

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None
        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        cv2.imshow("Game Boy Screen", image)
        cv2.waitKey(0)
        
    def get_area(self ,frame ):

        (cnts,hierarchy) = self.contours(frame)
        #print hierarchy
        Area = 0
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)

            # ensure that the approximated contour is "roughly" rectangular
            if len(approx) >= 4 and len(approx) <= 6:
                # compute the bounding box of the approximated contour and
                # use the bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)

                aspectRatio = w / float(h)

                # compute the solidity of the original contour
                area = cv2.contourArea(c)
                Area = max(area, Area)
                hullArea = cv2.contourArea(cv2.convexHull(c))
                solidity = area / float(hullArea)

                # compute whether or not the width and height, solidity, and
                # aspect ratio of the contour falls within appropriate bounds
                keepDims = w > 25 and h > 25
                keepSolidity = solidity > 0.9
                keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

                # ensure that the contour passes all our tests
                if keepDims and keepSolidity and keepAspectRatio:
                    # draw an outline around the target and update the status
                    # text
                    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                    status = "Target(s) Acquired"

                    # compute the center of the contour region and draw the
                    # crosshairs
                    M = cv2.moments(approx)
                    (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
                    (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
                    cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
                    cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)

        # draw the status text on the frame
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        # show the frame and record if a key is pressed
        if(self.debug):
            cv2.imshow("Frame", frame)

        return Area , frame

    def contours(self , frame):
        # convert the frame to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # find contours in the edge map
        (_,cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #_, cnts, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts , hierarchy

    def find_marker(self ,image):
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)

    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth
