import cv2


from threading import Thread
import numpy as np
import dlib
import urllib

import get_points
from helper import *
from glyphfunctions import *



class Webcam:
  
    def __init__(self , source , debug= False , root_url='192.168.43.1:8080'):
        self.source = source
        self.video_capture = cv2.VideoCapture(source)
        self.current_frame = self.video_capture.read()[1]
        self.retval = self.video_capture.read()[0]
        self.std = None
        self.debug = debug
        self.thread = None
        self.points = None
        self.tracker = None
        self.url = 'http://' + root_url + '/shot.jpg'
        self.name = 'webcam'
          
    # create thread for capturing images
    def start(self):
        #self.video_capture = cv2.VideoCapture(self.source)

        self.thread = True
        th=Thread(target=self._update_frame, args=()).start()
        if self.debug:
            print("Started threading , thread:",th)
    def get_IP_image(self):
        # Get our image from the phone
        imgResp = urllib.urlopen(self.url)

        # Convert our image to a numpy array so that we can work with it
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

        # Convert our image again but this time to opencv format
        img = cv2.imdecode(imgNp,-1)

        return img

    def stop(self):
        if self.debug:
            print ("video camera feed released")
        self.video_capture.release()
        #self.thread = False
    def set_url(self , ipaddress , port):
        self.url = 'http://' + ipaddress+port + '/shot.jpg'
  
    def _update_frame(self):
        while(self.thread):
            self.current_frame = self.video_capture.read()[1]
    def update_frame(self):
        if self.name == 'webcam':
            self.retval,self.current_frame = self.video_capture.read()
        elif self.name == 'ipcam':
            self.current_frame = cv2.imdecode(np.array(bytearray(self.video_capture.read()),dtype=np.uint8),-1)

    def set_source(self , name='webcam'):
        self.name = name
        if name == 'webcam':
            self.video_capture = cv2.VideoCapture(self.source)
        elif name == 'ipcam':
            self.video_capture = urllib.urlopen(self.url)
            if(self.debug):
                print "Ip address set to " , self.url
    def get_source(self):
        return self.name


    # get the current frame
    def get_current_frame(self):
        return self.retval,self.current_frame
    def to_gray(self,img):
        """
        Converts the input in grey levels
        Returns a one channel image
        """
        grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        return grey_img   
    
    def grey_histogram(img, nBins=64):
        """
        Returns a one dimension histogram for the given image
        The image is expected to have one channel, 8 bits depth
        nBins can be defined between 1 and 255 
        """

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
        if(self.debug):
            cv2.imshow('colorhist', h)
            cv2.waitKey(0)

        return h

    def extract_bright(self ,grey_img, histogram=False):
        """
        Extracts brightest part of the image.
        Expected to be the LEDs (provided that there is a dark background)
        Returns a Thresholded image
        histgram defines if we use the hist calculation to find the best margin
        """
        ## Searches for image maximum (brightest pixel)
        # We expect the LEDs to be brighter than the rest of the image
        #grey_img = cv.fromarray(grey_img)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_img)
        #print("Brightest pixel val is " ,maxVal ,"Dimmest pixel val is ", minVal)
    
        #We retrieve only the brightest part of the image
        # Here is use a fixed margin (80%), but you can use hist to enhance this one    
        if 0:
            ## Histogram may be used to wisely define the margin
            # We expect a huge spike corresponding to the mean of the background
            # and another smaller spike of bright values (the LEDs)
            hist = grey_histogram(img, nBins=64)
            [hminValue, hmaxValue, hminIdx, hmaxIdx] = cv.GetMinMaxHistValue(hist) 
            margin = 0# statistics to be calculated using hist data    
        else:  
            margin = 0.8
        
        thresh = int( maxVal * margin) # in pix value to be extracted
        #print ("Threshold is defined as " ,thresh)

        ret ,thresh_img = cv2.threshold(grey_img, maxVal - minVal , maxVal, cv2.THRESH_BINARY)
        
        return thresh_img

    def find_leds(self, thresh_img):
        """
        Given a binary image showing the brightest pixels in an image, 
        returns a result image, displaying found leds in a rectangle
        """
        try:
            contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        except:
            return None , None ,None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        regions = []
        i = 0
        cntrs = [] 
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)
            x,y,w,h = cv2.boundingRect(cnt)
            regions.append((x,y,w,h))
            #self.display_img(self.current_frame[x:x+w,y:y:h],str(i)  , delay = 10)
            i = i + 1
            print(self.current_frame[x:x+w,y:y+h])
            cntrs.append(self.current_frame[x:x+w,y:y:h])
        
        print cntrs    
        return thresh_img,regions,cntrs

    def leds_positions(self,regions):
        """
        Function using the regions in input to calculate the position of found leds
        """
        centers = []
        for x, y, width, height in regions:
            centers.append( [x+ (width / 2),y + (height / 2)])

        return centers
    
    def display_img(self,image,name , delay=1000):
        print ("size  , ",image.size)
        if (not image is None):
            cv2.imshow(name, image)
            cv2.waitKey(delay)

    def showStats(self,centers):

        if self.std == None:
            return

        self.std.clear()
        self.std.addstr(0,0,"Press Esc to Quit...")
        self.std.addstr(1,0,"Total number of Leds found :{} \t ".format((len(centers))))
        i = 2
        for c in centers:
            i=i+1
            self.std.addstr(i, 0, " X : {} \t {} \t".format(c[0],c[1]))
        

    
        self.std.refresh()

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
        print ("blas")
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

    def select_points(self):
        self.points = get_points.run(self.current_frame)
        self.tracker = dlib.correlation_tracker()
        points = self.points
        self.tracker.start_track(self.current_frame, dlib.rectangle(*points[0]))
        return points

    def get_points(self):
        return self.points

    def object_tracker(self, image):
        points = self.points
        self.tracker.update(image)
        rect = self.tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        print "Object tracked at [{}, {}] \r".format(pt1, pt2),
        cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)
        loc = (int(rect.left()), int(rect.top() - 20))
        txt = "Object tracked at [{}, {}]".format(pt1, pt2)
        cv2.putText(image, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        return image

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
