from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
from PIL import Image
from utils import letterbox_image, scale_coords
import numpy as np
import time

def detect_video(weights,webcam,img_size,conf_thres,iou_thres):

    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoCapture(webcam)
    fps = video.get(cv2.CAP_PROP_FPS)
    #print(fps)
    h = int(video.get(3))
    w = int(video.get(4))
    print(w,h)
    #h = 1280
    #w = 720
    result_video_filepath =  'webcam_yolov5_output.mp4'
    out  = cv2.VideoWriter(result_video_filepath,fourcc,int(fps),(h,w))

    yolov5_tflite_obj = yolov5_tflite(weights,img_size,conf_thres,iou_thres)

    size = (img_size,img_size)
    no_of_frames = 0
    try:
        while True:
        
            check, frame = video.read()
            
            if not check:
                break
            #frame = cv2.resize(frame,(h,w))
            #no_of_frames += 1
            image_resized = letterbox_image(Image.fromarray(frame),size)
            image_array = np.asarray(image_resized)

            normalized_image_array = image_array.astype(np.float32) / 255.0
            result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(normalized_image_array)
            
            if len(result_boxes) > 0:
                result_boxes = scale_coords(size,np.array(result_boxes),(w,h))
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                # org 
                org = (20, 40) 
                    
                # fontScale 
                fontScale = 0.5
                    
                # Blue color in BGR 
                color = (0, 255, 0) 
                    
                # Line thickness of 1 px 
                thickness = 1

                for i,r in enumerate(result_boxes):

                    org = (int(r[0]),int(r[1]))
                    cv2.rectangle(frame, (int(r[0]),int(r[1])), (int(r[2]),int(r[3])), (255,0,0), 1)
                    cv2.putText(frame, str(int(100*result_scores[i])) + '%  ' + str(result_class_names[i]), org, font,  
                                fontScale, color, thickness, cv2.LINE_AA)

            
            out.write(frame)
            
            cv2.imshow('output',frame)    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = time.time()
            print('FPS:',1/(end_time-start_time))
            start_time = end_time
        out.release()
    except:
        out.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('-wc','--webcam', type=int, default=0, help='webcam number 0,1,2 etc.') 
    parser.add_argument('--img_size', type=int, default=416, help='image size')  # height, width
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_video(opt.weights,opt.webcam,opt.img_size,opt.conf_thres,opt.iou_thres)