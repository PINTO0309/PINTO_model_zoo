import cv2
import time

print('opencv version:', cv2.__version__)

xml = 'saved_model.xml'
bin = 'saved_model.bin'
image = cv2.imread('test.png')
blob = cv2.dnn.blobFromImage(image, 1, (368, 368), (104, 117, 123))
net = cv2.dnn.readNetFromModelOptimizer(xml, bin)

### Dummy Pred ###########################################
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setInput(blob)
preds = net.forward()
### Dummy Pred ###########################################

print("[INFO] DNN_BACKEND_INFERENCE_ENGINE loading model...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] DNN_BACKEND_INFERENCE_ENGINE inference took " + str((end - start) * 1000) + " ms")


print("[INFO] DNN_TARGET_OPENCL_FP16 loading model...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] DNN_TARGET_OPENCL_FP16 inference took " + str((end - start) * 1000) + " ms")


print("[INFO] DNN_TARGET_CPU loading model...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] DNN_TARGET_CPU inference took " + str((end - start) * 1000) + " ms")


print("[INFO] DNN_TARGET_MYRIAD loading model...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] DNN_TARGET_MYRIAD inference took " + str((end - start) * 1000) + " ms")
