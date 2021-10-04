from unrealcv import client
import sys
import os
import numpy as np
import cv2
import io

camera_poses=np.array([[8131.582,2424.542,295.443,4.363,180.552,0.000],
[7515.561,2352.326,412.128,6.601,180.468,0.000],
[6811.178,2296.798,428.755,2.265,179.738,0.000],
[6238.702,2249.915,430.471,2.370,178.187,0.000],
[5853.423,2249.958,431.222,4.967,178.323,0.000],
[5335.302,2267.797,455.968,-1.181,180.208,0.000],
[4926.522,2267.869,436.683,1.036,180.758,0.000],
[4192.994,2231.891,458.685,-1.400,186.501,0.000],
[3518.811,2354.919,424.751,-2.552,183.908,0.000],
[2940.357,2315.401,399.961,2.014,183.513,0.000],
[2432.960,2293.505,413.523,4.014,182.941,0.000],
[1877.202,2268.354,443.451,0.414,180.008,0.000],
[1482.761,2268.300,446.299,3.031,181.440,0.000],
[1144.504,2259.795,464.217,3.031,181.440,0.000],
[715.544,2267.363,472.558,1.001,181.279,0.000],
[-155.427,2247.910,487.786,-0.413,179.976,0.000],
[-923.420,2248.823,478.974,-0.290,179.801,0.000],
[-1616.208,2251.235,470.393,-0.290,179.801,0.000],
[-2693.185,2254.983,457.054,-0.290,179.801,0.000],
[-3734.848,2267.302,451.761,0.602,178.032,0.000],
[-4077.070,2276.225,458.437,1.122,178.250,0.000],
[-4393.500,2311.624,465.688,0.590,130.252,0.000],
[-4623.441,2665.899,368.658,6.102,111.552,0.000],
[-4720.735,2953.385,411.037,6.937,99.380,0.000]])

fps = 10
times = np.arange(0,camera_poses.shape[0]*fps,fps)
filled_times = np.arange(0,camera_poses.shape[0]*fps)

filtered_poses = np.array([np.interp(filled_times, times, axis) for axis in camera_poses.T]).T

class UnrealcvStereo():

    def __init__(self):

        client.connect()
        if not client.isconnected():
            print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
            sys.exit(-1)

    def __str__(self):
        return client.request('vget /unrealcv/status')

    @staticmethod
    def set_position(pose):

        # Set position of the first camera
        client.request(f'vset /camera/0/location {pose[0]} {pose[1]} {pose[2]}')
        client.request(f'vset /camera/0/rotation {pose[3]} {pose[4]} {pose[5]}')

    @staticmethod
    def get_stereo_pair(eye_distance):
        res = client.request('vset /action/eyes_distance %d' % eye_distance)
        res = client.request('vget /camera/0/lit png')
        left = cv2.imdecode(np.frombuffer(res, dtype='uint8'), cv2.IMREAD_UNCHANGED)
        res = client.request('vget /camera/1/lit png')
        right = cv2.imdecode(np.frombuffer(res, dtype='uint8'), cv2.IMREAD_UNCHANGED)

        return left, right

    @staticmethod
    def convert_depth(PointDepth, f=320):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
        DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
        return PlaneDepth

    @staticmethod
    def get_depth():

        res = client.request('vget /camera/0/depth npy')
        point_depth = np.load(io.BytesIO(res))

        return UnrealcvStereo.convert_depth(point_depth)


    @staticmethod
    def color_depth(depth_map, max_dist):

        norm_depth_map = 255*(1-depth_map/max_dist)
        norm_depth_map[norm_depth_map < 0] =0
        norm_depth_map[depth_map == 0] =0

        return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)


if __name__ == '__main__':

    eye_distance = 10
    max_depth = 10
    stereo_generator = UnrealcvStereo()

    left_video = cv2.VideoWriter("left_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
    right_video = cv2.VideoWriter("right_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
    depth_folder = "depthmap"

    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    for i, pose in enumerate(filtered_poses):

        stereo_generator.set_position(pose)

        # Set the eye distance
        left, right = stereo_generator.get_stereo_pair(eye_distance)

        depth_map = stereo_generator.get_depth()

        depth_map[depth_map>max_depth] = max_depth
        depth_map_u16 = (depth_map*1000).astype(np.uint16)

        color_depth_map = stereo_generator.color_depth(depth_map, max_depth)
        left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

        # Save images
        left_video.write(left)

        right_video.write(right)
        cv2.imwrite(f"{depth_folder}/depth_frame{i:03d}.png",depth_map_u16)

        # Disp
        combined_image = np.hstack((left, right, color_depth_map))
        cv2.imshow("stereo", combined_image)

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

    left_video.release()
    right_video.release()
    cv2.destroyAllWindows()

