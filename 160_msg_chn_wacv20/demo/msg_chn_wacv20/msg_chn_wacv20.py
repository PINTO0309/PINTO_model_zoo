import cv2
import time
import numpy as np
from openvino.inference_engine import IECore
import os

class Msg_chn_wacv20():

    def __init__(self, model_path, device='CPU'):

        # Initialize model
        self.model = self.initialize_model(model_path, device)

    def __call__(self, image, sparse_depth, max_depth = 10.0):

        return self.estimate_depth(image, sparse_depth, max_depth)

    def initialize_model(self, model_path, device):

        self.ie = IECore()
        self.net = self.ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
        self.exec_net = self.ie.load_network(network=self.net, device_name=device, num_requests=2)

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def estimate_depth(self, image, sparse_depth, max_depth = 10.0):
        self.max_depth = max_depth

        input_rgb_tensor = self.prepare_rgb(image)
        input_depth_tensor = self.prepare_depth(sparse_depth)

        outputs = self.inference(input_rgb_tensor, input_depth_tensor)

        depth_map = self.process_output(outputs)

        return depth_map

    def prepare_rgb(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_height, self.img_width, self.img_channels = img.shape

        img_input = cv2.resize(img, (self.input_width,self.input_height))
        img_input = img_input/255
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis,:,:,:]

        return img_input.astype(np.float32)

    def prepare_depth(self, img):

        img_input = cv2.resize(img, (self.input_width,self.input_height), interpolation = cv2.INTER_NEAREST)

        img_input = img_input/self.max_depth*256
        img_input = img_input[np.newaxis,np.newaxis,:,:]

        return img_input.astype(np.float32)


    def inference(self, input_rgb_tensor, input_depth_tensor):
        start = time.time()
        outputs = self.exec_net.infer(
            inputs={
                self.rgb_input_name: input_rgb_tensor,
                self.depth_input_name:input_depth_tensor
            }
        )

        print(f'{1.0 / (time.time() - start)} FPS')
        return outputs

    def process_output(self, outputs):
        depth_map = np.squeeze(outputs[self.output_names[0]])/256*self.max_depth

        return cv2.resize(depth_map, (self.img_width, self.img_height))

    def getModel_input_details(self):

        key_iter = iter(self.net.input_info)
        self.depth_input_name = next(key_iter)
        self.rgb_input_name = next(key_iter)

        self.input_shape = self.net.input_info[self.depth_input_name].input_data.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def getModel_output_details(self):

        key_iter = iter(self.net.outputs)
        self.output_names = [
            next(key_iter),
            next(key_iter),
            next(key_iter),
        ]
        self.output_shape = self.net.outputs[self.output_names[0]].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

if __name__ == '__main__':

    from utils import make_depth_sparse, draw_depth, update_depth_density

    depth_density = 10
    depth_density_rate = 0.2
    max_depth = 10

    model_path='../models/saved_model_480x640/openvino/FP16/msg_chn_wacv20_480x640.xml'
    depth_estimator = Msg_chn_wacv20(model_path)

    cap_depth = cv2.VideoCapture("../outdoor_example/depthmap/depth_frame%03d.png", cv2.CAP_IMAGES)
    cap_rgb = cv2.VideoCapture("../outdoor_example/left_video.avi")

    while cap_rgb.isOpened() and cap_depth.isOpened():

        # Read frame from the videos
        ret, rgb_frame = cap_rgb.read()

        ret, depth_frame = cap_depth.read()

        if not ret:
            break

        depth_frame = depth_frame/1000 # to m

        # Make the depth map sparse
        depth_density, depth_density_rate = update_depth_density(depth_density, depth_density_rate, 1, 10)
        sparse_depth = make_depth_sparse(depth_frame, depth_density)

        # Fill the sparse depth map
        estimated_depth = depth_estimator(rgb_frame, sparse_depth)

        # Color depth maps
        color_gt_depth = draw_depth(depth_frame, max_depth)
        color_sparse_depth = draw_depth(sparse_depth, max_depth)
        color_estimated_depth = draw_depth(estimated_depth, max_depth)

        combined_img = np.vstack((np.hstack((rgb_frame, color_sparse_depth)),np.hstack((color_gt_depth,color_estimated_depth))))
        cv2.putText(combined_img,f'Density:{depth_density:.1f}%',(combined_img.shape[1]-500,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
        cv2.imshow("Estimated depth", combined_img)
        cv2.waitKey(1)
