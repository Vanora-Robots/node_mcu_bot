from ctypes import *
import math
import random

class Darknet_Yolo:
    def __init__(self):
        # lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
        self.lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict_p
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.make_boxes = self.lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype = POINTER(BOX)

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes = self.lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype = c_int

        self.make_probs = self.lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype = POINTER(POINTER(c_float))

        self.detect = self.lib.network_predict_p
        self.detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network_p
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.network_detect = self.lib.network_detect
        self.network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

    def sample(probs):
        s = sum(probs)
        probs = [a / s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs) - 1

    def c_array(ctype, values):
        return (ctype * len(values))(*values)

    def classify(self,net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self,net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        boxes = self.make_boxes(self.net)
        probs = self.make_probs(self.net)
        num = self.num_boxes(self.net)
        self.network_detect(self.net, im, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if probs[j][i] > 0:
                    res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def set_params(self , cfg = "cfg/tiny-yolo-voc.cfg" , weights="tiny-yolo-voc.weights" , data="cfg/voc.data"):
        self.net = self.load_net(cfg, weights,0)
        self.meta = self.load_meta(data)



class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]