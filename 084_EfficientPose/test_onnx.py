sess = rt.InferenceSession("model_float32_opt.onnx")
input_name = sess.get_inputs()[0].name

# resize    (test_frame is the result of cv2 videocapture)
f = min(368 / test_frame.shape[0], 368 / test_frame.shape[1])
test_frame = cv2.resize(test_frame, (0, 0), fx=f, fy=f)

test_frame = test_frame.astype('float32') / 255  # scale to range 0..1
test_frame = test_frame[..., ::-1]  # assumes channels last! ->  x = x[::-1, ...]  # BGR->RGB

# pad
blob = np.zeros((368,368,3), dtype='float32')  # channels last
d1 = int((368-test_frame.shape[0])/2)
d2 = int((368-test_frame.shape[1])/2)
blob[d1:368-d1,d2:368-d2] = test_frame

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for i in range(3):
    blob[..., i] -= mean[i]
    blob[..., i] /= std[i]

out = sess.run(None, {input_name: [blob]})