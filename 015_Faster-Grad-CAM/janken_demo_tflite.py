import cv2
import time
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

model_path = "model/" 

if os.path.exists(model_path):
    # load csv 
    print("csv loading...")
    channel_weight = np.loadtxt(model_path + "channel_weight.csv", delimiter=",")
    channel_adress = np.loadtxt(model_path + "channel_adress.csv", delimiter=",", dtype="float")
    channel_adress = channel_adress.astype(np.int)
    vector_pa = np.loadtxt(model_path + "vector_pa.csv", delimiter=",")
    kmeans = joblib.load(model_path + "k-means.pkl.cmp")

else:
    print("Nothing model folder")

def get_score_arc(pa_vector, test):
    # cosine similarity
    cos_similarity = cosine_similarity(test, pa_vector)

    return np.max(cos_similarity)

def cosine_similarity(x1, x2): 
    if x1.ndim == 1:
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim

def predict_faster_gradcam(channel, vector, img, kmeans, channel_weight, channel_adress):
    channel_out = channel[0]
    
    # k-means and heat_map
    cluster_no = kmeans.predict(vector)
    cam = np.dot(channel_out[:,:,channel_adress[cluster_no[0]]], channel_weight[cluster_no[0]])

    # nomalize
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

def get_x_y_limit(heatmap, thresh):
    map_ = np.where(heatmap>thresh)
    x_max = np.max(map_[1])
    x_min = np.min(map_[1])
    y_max = np.max(map_[0])
    y_min = np.min(map_[0])

    x_max = int(x_max)
    x_min = int(x_min)
    y_max = int(y_max)
    y_min = int(y_min)
    return x_min, y_min, x_max, y_max

def bounding_box(img, x_min, y_min, x_max, y_max):
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
    return img

def main():
    camera_width =  352
    camera_height = 288
    input_size = 96
    hand_thresh = 0.25
    OD_thresh = 0.8
    fps = ""
    message1 = "Push [q] to quit."
    message2 = "Push [s] to change mode."
    hand = ""
    elapsedTime = 0
    like_OD = False # like object detection
    result = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    interpreter = Interpreter(model_path=model_path + "weights_weight_quant.tflite")
    try:
        interpreter.set_num_threads(4)
    except:
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ret, image = cap.read()
        image = image[:,32:320]
        if not ret:
            break

        img = cv2.resize(image, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        channel_out = interpreter.get_tensor(output_details[0]['index'])
        test_vector = interpreter.get_tensor(output_details[1]['index'])

        score = get_score_arc(vector_pa, test_vector)

        if score < hand_thresh: # hand is gu
            hand = "gu"
            color = (255, 0, 0)
            heatmap = predict_faster_gradcam(channel_out, test_vector, image, kmeans, channel_weight, channel_adress)
            if like_OD:
                x_min, y_min, x_max, y_max = get_x_y_limit(heatmap, OD_thresh)
                result = bounding_box(image, x_min, y_min, x_max, y_max)
            else:
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                image = np.copy(cv2.addWeighted(heatmap, 0.5, image, 0.5, 2.2))

        else: # hand is pa
            hand = "pa"
            color = (0, 0, 255)

        # message
        cv2.putText(image, "{0} {1:.1f} Score".format(hand, score),(camera_width - 290, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        cv2.putText(image, message1, (camera_width - 285, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, message2, (camera_width - 285, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, fps, (camera_width - 175, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 0 ,0), 1, cv2.LINE_AA)

        cv2.imshow("Result", image)

        # FPS
        elapsedTime = time.time() - t1
        fps = "{:.0f} FPS".format(1/elapsedTime)

        # quit or change mode
        key = cv2.waitKey(1)&0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if like_OD == True:
                like_OD = False
            else:
                like_OD = True

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
