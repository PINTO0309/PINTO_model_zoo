
import cv2
import random
import numpy as np
import math
import os

class BBox:

    def __init__(self, label, xyrb, score=0, landmark=None, rotate = False):

        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate
        #避免出现rb小于xy的时候
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join([str(item[:2]) for item in self.landmark]) if self.landmark is not None else "empty"
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x-1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y-1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x-1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y-1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou

def intv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def floatv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([float(item) for item in value])
    elif isinstance(value, list):
        return [float(item) for item in value]
    elif value is None:
        return 0
    else:
        return float(value)


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low


def mkdirs_from_file_path(path):

    try:
        path = path.replace("\\", "/")
        p0 = path.rfind('/')
        if p0 != -1:
            path = path[:p0]

            if not os.path.exists(path):
                os.makedirs(path)

    except Exception as e:
        print(e)


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # # image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # # return image[:,:,(2,1,0)]
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image


def imwrite(path, image):
    path = path.replace("\\", "/")
    mkdirs_from_file_path(path)

    suffix = path[path.rfind("."):]
    ok, data = cv2.imencode(suffix, image)

    if ok:
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(e)
    return False



class RandomColor(object):

    def __init__(self, num):
        self.class_mapper = {}
        self.build(num)
        

    def build(self, num):

        self.colors = []
        for i in range(num):
            c = (i / (num + 1) * 360, 0.9, 0.9)
            t = np.array(c, np.float32).reshape(1, 1, 3)
            t = (cv2.cvtColor(t, cv2.COLOR_HSV2BGR) * 255).astype(np.uint8).reshape(3)
            self.colors.append(intv(tuple(t)))
        
        seed = 0xFF01002
        length = len(self.colors)
        for i in range(length):
            a = i
            seed = (((i << 3 ) + 3512301) ^ seed) & 0x0FFFFFFF
            b = seed % length
            x = self.colors[a]
            y = self.colors[b]
            self.colors[a] = y
            self.colors[b] = x

    def get_index(self, label):
        if isinstance(label, int):
            return label % len(self.colors)
        elif isinstance(label, str):
            if label not in self.class_mapper:
                self.class_mapper[label] = len(self.class_mapper)
            return self.class_mapper[label]
        else:
            raise Exception("label is not support type{}, must be str or int".format(type(label)))

    def __getitem__(self, label):
        return self.colors[self.get_index(label)]


_rand_color = None
def randcolor(label, num=32):
    global _rand_color

    if _rand_color is None:
        _rand_color = RandomColor(num)
    return _rand_color[label]



#(239, 121, 162)
def drawbbox(image, bbox, color=None, thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255)):

    if color is None:
        color = randcolor(bbox.label)

    #text = f"{bbox.label} {bbox.score:.2f}"
    text = f"{bbox.score:.2f}"
    x, y, r, b = intv(bbox.box)
    w = r - x + 1
    h = b - y + 1

    cv2.rectangle(image, (x, y, r-x+1, b-y+1), color, thickness, 16)

    border = thickness / 2
    pos = (x + 3, y - 5)
    cv2.rectangle(image, intv(x - border, y - 21, w + thickness, 21), color, -1, 16)
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

    if bbox.haslandmark:
        for i in range(len(bbox.landmark)):
            x, y = bbox.landmark[i][:2]
            cv2.circle(image, intv(x, y), 3, landmarkcolor, -1, 16)


def pad(image, stride=32):

    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image


def log(v):

    if isinstance(v, tuple) or isinstance(v, list) or isinstance(v, np.ndarray):
        return [log(item) for item in v]
    
    base = np.exp(1)
    if abs(v) < base:
        return v / base
    
    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)
    
def exp(v):

    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def file_name_no_suffix(path):
    path = path.replace("\\", "/")

    p0 = path.rfind("/") + 1
    p1 = path.rfind(".")

    if p1 == -1:
        p1 = len(path)
    return path[p0:p1]


def file_name(path):
    path = path.replace("\\", "/")
    p0 = path.rfind("/") + 1
    return path[p0:]