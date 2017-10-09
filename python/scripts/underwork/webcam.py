import cv2


from threading import Thread
import numpy as np
import dlib
import urllib





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
    


    def select_points(self):
        self.points = get_points.run(self.current_frame)
        self.tracker = dlib.correlation_tracker()
        points = self.points
        self.tracker.start_track(self.current_frame, dlib.rectangle(*points[0]))
        return points

    def get_points(self):
        return self.points



    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img
