import numpy as np
import math
import time
import sys
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter
import cv2

LABELS = [
'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
'fire hydrant','???','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
'cow','elephant','bear','zebra','giraffe','???','backpack','umbrella','???','???',
'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
'skateboard','surfboard','tennis racket','bottle','???','wine glass','cup','fork','knife','spoon',
'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
'cake','chair','couch','potted plant','bed','???','dining table','???','???','toilet',
'???','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
'sink','refrigerator','???','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

class ObjectDetectorLite():
    def __init__(self, model_path='detect.tflite'):
        self.interpreter = Interpreter(model_path=model_path)
        try:
            self.interpreter.set_num_threads(4)
        except:
            pass
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.max_detections = 10
        self.non_max_suppression_score_threshold = 0.7
        self.intersection_over_union_threshold = 0.6
        self.y_scale = 10.0
        self.x_scale = 10.0
        self.h_scale = 5.0
        self.w_scale = 5.0
        self.anchors = np.load('./anchors.npy')


    def decode_box_encodings(self, box_encoding, anchors):
        num_boxes = box_encoding.shape[0]
        decoded_boxes = np.zeros((num_boxes, 4), dtype=np.float32)
        for i in range(num_boxes):
            ycenter = box_encoding[i, 0] / self.y_scale * anchors[i, 2] + anchors[i, 0]
            xcenter = box_encoding[i, 1] / self.x_scale * anchors[i, 3] + anchors[i, 1]
            half_h = 0.5 * math.exp((box_encoding[i, 2] / self.h_scale)) * anchors[i, 2]
            half_w = 0.5 * math.exp((box_encoding[i, 3] / self.w_scale)) * anchors[i, 3]
            decoded_boxes[i, 0] = (ycenter - half_h) # ymin
            decoded_boxes[i, 1] = (xcenter - half_w) # xmin
            decoded_boxes[i, 2] = (ycenter + half_h) # ymax
            decoded_boxes[i, 3] = (xcenter + half_w) # xmax
        return decoded_boxes


    def non_maximum_suprression(self, box_encoding, class_predictions):
        val, idx = class_predictions[:, 1:].max(axis=1), \
                   class_predictions[:, 1:].argmax(axis=1)
        thresh_val, thresh_idx = np.array(val)[val>=self.non_max_suppression_score_threshold], \
                                 np.array(idx)[val>=self.non_max_suppression_score_threshold]
        thresh_box = np.array(box_encoding)[val>=self.non_max_suppression_score_threshold]
        anchor_count = thresh_box.shape[0]
        thresh_box_stack = np.hstack((thresh_box, thresh_idx[:, np.newaxis], thresh_val[:, np.newaxis]))
        thresh_box_stack = thresh_box_stack[np.argsort(thresh_box_stack[:, 5])[::-1]]
        active_box_candidate = np.ones((anchor_count, 1))
        thresh_box_stack = np.hstack((thresh_box_stack, active_box_candidate))
        box_detected_flg = np.zeros((anchor_count, 1))
        thresh_box_stack = np.hstack((thresh_box_stack, box_detected_flg))
        num_boxes_kept = anchor_count
        num_active_candidate = anchor_count
        output_size = min(num_active_candidate, self.max_detections)
        num_selected_count = 0

        for i in range(num_boxes_kept):
            if (num_active_candidate == 0 or num_selected_count >= output_size):
                break
            if (thresh_box_stack[i, 6] == 1):
                thresh_box_stack[i, 6] = 0
                thresh_box_stack[i, 7] = 1
                num_active_candidate -= 1
                num_selected_count += 1
            else:
                continue

            # thresh_box_stack = [ymin, xmin, ymax, xmax, class_idx, prob]
            for j in range(i + 1, num_boxes_kept):
                if (thresh_box_stack[j, 6] == 1):
                    intersection_over_union = self.compute_intersection_over_union(thresh_box_stack[i], thresh_box_stack[j])
                    if (intersection_over_union > self.intersection_over_union_threshold):
                        thresh_box_stack[j, 6] = 0
                        num_active_candidate -= 1

        return thresh_box_stack[thresh_box_stack[:, 7] == 1, :6] #[ymin, xmin, ymax, xmax, class_idx, prob]


    def compute_intersection_over_union(self, box_i, box_j):
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
        if (area_i <= 0 or area_j <= 0):
            return 0.0
        intersection_ymin = max(box_i[0], box_j[0])
        intersection_xmin = max(box_i[1], box_j[1])
        intersection_ymax = min(box_i[2], box_j[2])
        intersection_xmax = min(box_i[3], box_j[3])
        intersection_area = max(intersection_ymax - intersection_ymin, 0.0) * max(intersection_xmax - intersection_xmin, 0.0)
        return intersection_area / (area_i + area_j - intersection_area)


    def detect(self, image):
        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        frame = frame - 127.5
        frame = frame * 0.007843

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count = self.interpreter.get_tensor(self.output_details[3]['index'])[0]

        #decoded_boxes = self.decode_box_encodings(boxes, self.anchors)
        #detected_boxes = self.non_maximum_suprression(decoded_boxes, classes)

        #return detected_boxes #[ymin, xmin, ymax, xmax, class_idx, prob]
        return boxes, classes, scores, count


if __name__ == '__main__':
    start_time = time.perf_counter()

    detector = ObjectDetectorLite('03_integer_quantization/ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite')
    #detector = ObjectDetectorLite('03_integer_quantization/ssdlite_mobilenet_v2_coco_300_integer_quant.tflite')
    #detector = ObjectDetectorLite('02_weight_quantization/ssdlite_mobilenet_v2_coco_300_weight_quant.tflite')
    image = cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB)
    image_height = image.shape[0]
    image_width  = image.shape[1]
    results = detector.detect(image)
    print(results)

    stop_time = time.perf_counter()
    print("time: ", stop_time - start_time)
    sys.exit(0)

    for box in results:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        classnum = int(box[4])
        probability = box[5]
        print('coordinates: ({}, {})-({}, {}). class: "{}". probability: {:.2f}'.format(xmin, ymin, xmax, ymax, classnum, probability))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, '{}: {:.2f}'.format(LABELS[classnum],probability), (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    stop_time = time.perf_counter()
    print("time: ", stop_time - start_time)

    cv2.imwrite('result.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

