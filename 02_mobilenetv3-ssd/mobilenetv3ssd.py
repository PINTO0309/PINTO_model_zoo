import numpy as np
import time
import math
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter
import cv2

LABELS = [
'???','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','???','stop sign','parking meter','bench','bird','cat','dog','horse',
'sheep','cow','elephant','bear','zebra','giraffe','???','backpack','umbrella','???',
'???','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','???','wine glass','cup','fork','knife',
'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
'donut','cake','chair','couch','potted plant','bed','???','dining table','???','???',
'toilet','???','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
'toaster','sink','refrigerator','???','book','clock','vase','scissors','teddy bear','hair drier',
'toothbrush']

y_scale = 10.0
x_scale = 10.0
h_scale = 5.0
w_scale = 5.0

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

    def decode_box_encodings(self, box_encoding, anchors, num_boxes):
        decoded_boxes = np.zeros((1, num_boxes, 4), dtype=np.float32)
        for i in range(num_boxes):
            ycenter = box_encoding[0][i][0] / y_scale * anchors[i][2] + anchors[i][0]
            xcenter = box_encoding[0][i][1] / x_scale * anchors[i][3] + anchors[i][1]
            half_h = 0.5 * math.exp((box_encoding[0][i][2] / h_scale)) * anchors[i][2]
            half_w = 0.5 * math.exp((box_encoding[0][i][3] / w_scale)) * anchors[i][3]
            decoded_boxes[0][i][0] = (ycenter - half_h) # ymin
            decoded_boxes[0][i][1] = (xcenter - half_w) # xmin
            decoded_boxes[0][i][2] = (ycenter + half_h) # ymax
            decoded_boxes[0][i][3] = (xcenter + half_w) # xmax
        return decoded_boxes

    #def _boxes_coordinates(self,
    #                        image,
    #                        boxes,
    #                        classes,
    #                        scores,
    #                        max_boxes_to_draw=20,
    #                        min_score_thresh=.5):
    #    if not max_boxes_to_draw:
    #        max_boxes_to_draw = boxes.shape[0]
    #    number_boxes = min(max_boxes_to_draw, boxes.shape[0])
    #    person_boxes = []
    #    for i in range(number_boxes):
    #        if scores is None or scores[i] > min_score_thresh:
    #            box = tuple(boxes[i].tolist())
    #            ymin, xmin, ymax, xmax = box
    #            im_height, im_width, _ = image.shape
    #            left, right, top, bottom = [int(z) for z in (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)]
    #            person_boxes.append([(left, top), (right, bottom), scores[i], LABELS[classes[i]]])
    #    return person_boxes


    def detect(self, image, threshold=0.1):
        # Resize and normalize image for network input
        frame = cv2.resize(image, (320, 320))
        #frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        #frame = frame.astype(np.uint8)
        print(frame.dtype)
        #frame = frame - 127.5
        #frame = frame * 0.007843

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()
        print("time: ", stop_time - start_time)

        # get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        #scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        #num = self.interpreter.get_tensor(self.output_details[3]['index'])

        print("boxes=", boxes[0])
        print("boxes.shape=", boxes[0].shape)
        np.savetxt('./boxes.csv', boxes[0], delimiter=',')

        print("classes=", classes[0])
        print("classes.shape=", classes[0].shape)
        np.savetxt('./classes.csv', classes[0], delimiter=',')

        box_classe = np.argmax(classes[0], axis=1)
        print("box_classe=", box_classe)
        print("box_classe.shape=", box_classe.shape)
        np.savetxt('./box_classe.csv', box_classe, delimiter=',')

        #print("scores=", scores[0])
        #print("scores.shape=", scores[0].shape)
        #np.savetxt('./scores.csv', scores[0], delimiter=',')

        # Find detected boxes coordinates
        #return self._boxes_coordinates(image,
        #                    np.squeeze(boxes[0]),
        #                    np.squeeze(classes[0]+1).astype(np.int32),
        #                    np.squeeze(scores[0]),
        #                    min_score_thresh=threshold)

        return None

if __name__ == '__main__':
    #detector = ObjectDetectorLite('./01_mobilenetv3_small/01_coco/02_weight_quantization/ssd_mobilenet_v3_small_coco_weight_quant.tflite')
    detector = ObjectDetectorLite('./01_mobilenetv3_small/01_coco/03_integer_quantization/ssd_mobilenet_v3_small_coco_integer_quant.tflite')

    image = cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB)

    result = detector.detect(image, 0.4)
    print(result)

    for obj in result:
        print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.format(obj[0], obj[1], obj[3], obj[2]))

        cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
        cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]), (obj[0][0], obj[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imwrite('result.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

