import urllib
import numpy as np
import cv2

class Ipcam:
    #We init the IP cam with the client IP address
    def __init__(self , root_url , debug = False):
        self.url = 'http://' + root_url + '/shot.jpg'
        self.image = None
        self.debug = debug


    def get_IP_image(self):
        # Get our image from the ipcam
        try:
            if self.debug:
                print ("Ip cam " , self.url)

            imgResp = urllib.urlopen(self.url)
            
        except urllib.error.HTTPError as err:
            print "Falied to retreav from IP"
            print(err.code)
            return None

        # Convert our image to a numpy array so that we can work with it
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

        # Convert our image again but this time to opencv format
        img = cv2.imdecode(imgNp,-1)
        self.image = img

        if self.debug:
            cv2.namedWindow("IP cam", cv2.WINDOW_NORMAL)
            cv2.imshow("IP cam", self.image)

        return img


