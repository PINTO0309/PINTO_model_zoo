import numpy as np
import math
import time
import sys
import cv2
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

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

if __name__ == '__main__':
    image = cv2.imread('dog.jpg')
    #interpreter = Interpreter(model_path='ssd_mobilenet_v3_small_coco_weight_quant_postprocess.tflite')
    interpreter = Interpreter(model_path='ssd_mobilenet_v3_small_coco_integer_quant_postprocess.tflite')
    try:
        interpreter.set_num_threads(4)
    except:
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start_time = time.perf_counter()
    t1 = time.perf_counter()

    image_height = image.shape[0]
    image_width  = image.shape[1]

    # Resize and normalize image for network input
    t3 = time.perf_counter()
    frame = cv2.resize(image, (320, 320))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.float32)
    frame = frame - 127.5
    frame = frame * 0.007843
    t4 = time.perf_counter()
    print("resize and normalize time: ", t4 - t3)

    # run model
    t5 = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    t6 = time.perf_counter()
    print("inference + postprocess time: ", t6 - t5)

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = interpreter.get_tensor(output_details[3]['index'])[0]

    #print(boxes, classes, scores, count)

    stop_time = time.perf_counter()
    print("TOTAL time: ", stop_time - start_time)
    print(boxes, classes, scores, count)
    #sys.exit(0)

    for box, classidx, score in zip(boxes, classes, scores):
        probability = score
        if probability >= 0.50:
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)
            classnum = int(classidx)
            probability = score
            print('coordinates: ({}, {})-({}, {}). class: "{}". probability: {:.2f}'.format(xmin, ymin, xmax, ymax, classnum, probability))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(LABELS[classnum],probability), (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    stop_time = time.perf_counter()
    print("time: ", stop_time - start_time)

    cv2.imwrite('result.jpg', image)

