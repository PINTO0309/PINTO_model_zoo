from openvino.inference_engine import IECore
import numpy as np
import cv2
import pprint

XML_PATH = "openvino/FP16/movenet_singlepose_thunder_3.xml"
BIN_PATH = "openvino/FP16/movenet_singlepose_thunder_3.bin"

ie = IECore()
net = ie.read_network(model=XML_PATH, weights=BIN_PATH)
input_blob = next(iter(net.input_info))
exec_net = ie.load_network(net, device_name='CPU', num_requests=1)
inference_request = exec_net.requests[0]

img = cv2.imread('test.png')
img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img)
img = img.astype(np.float32)
img = img[np.newaxis,:,:,:]

exec_net.infer(inputs={input_blob: img})
pprint.pprint(inference_request.output_blobs)
output = inference_request.output_blobs['Identity'].buffer

pprint.pprint(output)
