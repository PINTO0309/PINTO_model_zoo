from yolov5_tflite_image_inference import detect_image
import argparse
from glob import glob
import os


def detect_from_folder_of_images(weights,folder_path,img_size,conf_thres,iou_thres):

    for file in glob(os.path.join(folder_path,'*')):
        print('Processing ',file,' now ...')
        
        detect_image(weights,file,img_size,conf_thres,iou_thres)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('-f','--folder_path', type=str,required=True, help='folder path')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=416, help='image size')  # height, width
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_from_folder_of_images(opt.weights,opt.folder_path,opt.img_size,opt.conf_thres,opt.iou_thres)