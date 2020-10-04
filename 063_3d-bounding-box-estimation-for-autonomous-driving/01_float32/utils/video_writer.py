import argparse
import cv2
import os
import numpy as np

def write_video(args):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(os.path.join(args.out, args.name), fourcc, args.fps, (1557, 394))

    for image in sorted(os.listdir(args.path)):
        img = cv2.imread(os.path.join(args.path, image))
        h, w = img.shape[0], img.shape[1]

        video_writer.write(np.uint8(img))


def main():
    parser = argparse.ArgumentParser(description='Make video from image frames',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, default='../result_vgg_0093', help='Image folder')
    parser.add_argument('-n', '--name', type=str, default='KITTI_3d_mobinenet_0093.avi', help='Output video name')
    parser.add_argument('-f', '--fps', type=int, default=15, help='Video fps')
    parser.add_argument('-o', '--out', type=str, default='../', help='Output folder')

    args = parser.parse_args()

    write_video(args)

if __name__ == '__main__':
    main()