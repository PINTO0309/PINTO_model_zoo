import os
import cv2
import copy
import glob
import shutil
import argparse
import numpy as np
import torch
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lib.preprocess import h36m_coco_format
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def wrap(func, *args, unsqueeze=False):
	args = list(args)
	for i, arg in enumerate(args):
		if type(arg) == np.ndarray:
			args[i] = torch.from_numpy(arg)
			if unsqueeze:
				args[i] = args[i].unsqueeze(0)

	result = func(*args)

	if isinstance(result, tuple):
		result = list(result)
		for i, res in enumerate(result):
			if type(res) == torch.Tensor:
				if unsqueeze:
					res = res.squeeze(0)
				result[i] = res.numpy()
		return tuple(result)
	elif type(result) == torch.Tensor:
		if unsqueeze:
			result = result.squeeze(0)
		return result.numpy()
	else:
		return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))

def show2Dpose(kps, img):
    connections = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
        [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
    ]
    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)
    return img

def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(figure_path, output_dir, file_name, args, with_norm):
    # Genarate 2D pose
    keypoints, scores = hrnet_pose(figure_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    ## 3D
    img = cv2.imread(figure_path)
    img_size = img.shape

    # keypoints: (1, 1, 17, 2)
    # input_2D_no: (1, 17, 2)
    input_2D_no = keypoints[0,:,:,:]
    if not with_norm:
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])
    else:
        input_2D = input_2D_no
    input_2D = input_2D.astype(np.float32)

    # Load model
    providers = []
    if args.provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.provider == 'tensorrt':
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    onnx_session = onnxruntime.InferenceSession(
        path_or_bytes=args.model,
        providers=providers,
    )
    if not with_norm:
        input_name1: str = onnx_session.get_inputs()[0].name
    else:
        input_name1: str = onnx_session.get_inputs()[0].name
        input_name2: str = onnx_session.get_inputs()[1].name
        input_name3: str = onnx_session.get_inputs()[2].name

    # Inference
    if not with_norm:
        output_3D = \
            onnx_session.run(
                None,
                {input_name1: input_2D},
            )[0]
    else:
        output_3D = \
            onnx_session.run(
                None,
                {
                    input_name1: input_2D,
                    input_name2: np.asarray(img_size[0]),
                    input_name3: np.asarray(img_size[1]),
                },
            )[0]


    rot =  np.asarray(
        [
            0.1407056450843811,
            -0.1500701755285263,
            -0.755240797996521,
            0.6223280429840088
        ],
        dtype=np.float32
    )
    post_out = camera_to_world(output_3D[0], R=rot, t=0)
    post_out[..., 2] -= np.min(post_out[..., 2])

    ## 2D
    image = show2Dpose(input_2D_no[0], copy.deepcopy(img))

    ## 3D
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose( post_out, ax)

    output_dir_3D = f'{output_dir}/pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    plt.savefig(f'{output_dir_3D}_3D.png', dpi=200, format='png', bbox_inches = 'tight')

    ## all
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    for i in range(len(image_3d_dir)):
        image_2d = image
        image_3d = plt.imread(image_3d_dir[i])
        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]
        ## show
        font_size = 12
        ax = plt.subplot(121)
        showimage(ax, image_2d[..., ::-1])
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Pose", fontsize = font_size)

        ## save
        plt.savefig(f'{output_dir}/{file_name}_pose.png', dpi=200, bbox_inches = 'tight')

    shutil.rmtree(f'{output_dir}/pose3D')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mod',
        '--model',
        type=str,
        default='htnet_1x17x2_with_norm.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    args = parser.parse_args()
    with_norm = '_with_norm' in args.model
    items = glob.glob('*.jpg')
    for i, file_name in enumerate(items):
        print(f'Gnenerate Pose For {file_name}')
        output_dir = 'output'
        get_pose3D(file_name, output_dir, file_name[:-4], args, with_norm)

