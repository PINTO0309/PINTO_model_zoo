import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#===============================================================================================
import tensorflow_hub as hub
import numpy as np
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")

from PIL import Image
images = [np.array(Image.open('dog_320x320.jpg'))]
boxes, scores, classes, num_detections = detector(images)
print('===================================== tfhub')
print(boxes)
print(scores)
print(classes)
print(num_detections)

#===============================================================================================
LABELS = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

import cv2
import pprint
from openvino.inference_engine import IECore
ie = IECore()
model = 'openvino/FP16/model'
net = ie.read_network(model=f'{model}.xml', weights=f'{model}.bin')
exec_net = ie.load_network(network=net, device_name='CPU')

img = cv2.imread('dog.jpg')
w = int(img.shape[0])
h = int(img.shape[1])

scale_w = w / 320
scale_h = h / 320

resized_frame = cv2.resize(img, (320, 320))
resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
resized_frame = resized_frame[np.newaxis,:,:,:].transpose((0,3,1,2))
print('@@@@@@@@@@@@@@@@@@@@@ resized_frame.shape', resized_frame)
outputs = exec_net.infer(inputs={'serving_default_images:0': resized_frame})
print('===================================== openvino')
# pprint.pprint(outputs)

print(outputs['StatefulPartitionedCall:3'])
print(outputs['StatefulPartitionedCall:3'] * 320)

bboxes = outputs['StatefulPartitionedCall:3'] * 320
bboxes = np.where(bboxes < 0.0, 0.0, bboxes)

print('Slice__1691/Split.0', outputs['Slice__1691/Split.0'].shape)
print('StatefulPartitionedCall:1', outputs['StatefulPartitionedCall:1'].shape)
print('StatefulPartitionedCall:2', outputs['StatefulPartitionedCall:2'].shape)
print('StatefulPartitionedCall:3', outputs['StatefulPartitionedCall:3'].shape)

# bbox = [ymin, xmin, ymax, xmax]

box = bboxes[0][0]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

box = bboxes[0][1]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

box = bboxes[0][2]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

cv2.imwrite('dog_result.jpg', img)

#===============================================================================================
import tensorflow as tf
import pprint
import os

def structure_print():
    print('')
    print(f'model: {os.path.basename(model_tflite)}')
    print('')
    print('==INPUT============================================')
    pprint.pprint(interpreter.get_input_details())
    print('')
    print('==OUTPUT===========================================')
    pprint.pprint(interpreter.get_output_details())

model_tflite = 'model_float32.tflite'
interpreter = tf.lite.Interpreter(model_tflite, num_threads=4)
interpreter.allocate_tensors()
structure_print()

in_frame = cv2.resize(img, (320, 320))
in_frame = in_frame.reshape((1, 320, 320, 3))
input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, in_frame.astype(np.float32))
interpreter.invoke()

bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
confs = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])

print(bboxes.shape)
print(bboxes)
print(class_ids.shape)
print(class_ids) # We need to add +1 to the index of the result.
print(confs.shape)
print(confs)

box = bboxes[0][0]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

box = bboxes[0][1]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

box = bboxes[0][2]
cv2.rectangle(img, (int(box[1] * scale_h), int(box[0] * scale_w)), (int(box[3] * scale_h), int(box[2] * scale_w)), (0,255,0), 2, 16)

cv2.imwrite('dog_result_tflite.jpg', img)