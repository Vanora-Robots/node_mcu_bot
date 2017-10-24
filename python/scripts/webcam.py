import cv2
from threading import Thread
import glob





class Webcam:
  
    def __init__(self , debug= False , name = 'webcam'  ):
        self.debug = debug
        self.name = name
        self.source_list = self.get_source_list()
        self.source = None
        self.status = True

    def get_source_list(self, path='/dev/video*'):
        self.path = path
        return glob.glob(path)

    # create thread for capturing images
    def start(self):
        #self.video_capture = cv2.VideoCapture(self.source)

        self.thread = True
        th=Thread(target=self._update_frame, args=()).start()
        if self.debug:
            print("Started threading , thread:",th)

    def set_status(self, status):
        self.status = status
    def get_status(self):
        return self.status

    def stop(self):
        if self.debug:
            print ("video camera feed released")
        self.video_capture.release()
        #self.thread = False

  
    def _update_frame(self):
        while(self.thread):
            self.current_frame = self.video_capture.read()[1]
    def update_frame(self):
        if self.name == 'webcam':
            self.retval,self.current_frame = self.video_capture.read()
        elif self.name == 'ipcam':
            self.current_frame = cv2.imdecode(np.array(bytearray(self.video_capture.read()),dtype=np.uint8),-1)

    def set_source(self , source):
        self.path =  self.path.replace('*','')
        self.source = int(self.source_list[source].replace(self.path, ""))
        if(self.debug):
            print self.path
            print self.source_list[source]
            print ("Source picked to be " + str(self.source))
        self.video_capture = cv2.VideoCapture(self.source)


    def get_source(self):
        return self.name


    # get the current frame
    def get_current_frame(self):
        return self.retval,self.current_frame





