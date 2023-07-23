import random
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class get_simcc_maximum_Layer(nn.Module):
    def __init__(self, model_input_width, model_input_height):
        super(get_simcc_maximum_Layer, self).__init__()
        self.model_input_width = model_input_width
        self.model_input_height = model_input_height

    def forward(self, simcc_x, simcc_y, bboxes_width_height):
        N, K, Wx = simcc_x.shape

        # get maximum value locations
        x_locs = torch.argmax(simcc_x, dim=2, keepdim=True)
        y_locs = torch.argmax(simcc_y, dim=2, keepdim=True)
        locs = torch.cat((x_locs, y_locs), dim=2).to(torch.float32)

        """
        max_val_x.shape
            torch.Size([133])
        max_val_y.shape
            torch.Size([133])
        """
        max_val_x = torch.amax(simcc_x, dim=2, keepdim=True)
        max_val_y = torch.amax(simcc_y, dim=2, keepdim=True)

        # get maximum value across x and y axis
        vals = torch.minimum(max_val_x, max_val_y)
        locs = locs / 2.0

        # scale conversion
        """
        bbox: x1, y1, x2, y2
            array([  0,   0, 218, 346])
        center
            array([109., 173.])
        scale -> bbox x 1.25
            array([272.5, 432.5])
        """

        """
        keypoints.shape
            (1, 133, 2)
        model_input_size
            (192, 256)
        scale
            array([324.375, 432.5  ])
        center
            array([109., 173.])
        """
        bboxes_width_height = torch.unsqueeze(bboxes_width_height, dim=1)
        scale = bboxes_width_height * 1.25
        w = scale[..., 0:1]
        h = scale[..., 1:2]
        w_div_075 = w / 0.75
        h_mul_075 = h * 0.75
        w_scaled = torch.maximum(h_mul_075, w)
        h_scaled = torch.maximum(w_div_075, h)
        scale = torch.cat([w_scaled, h_scaled], dim=2)

        scale = scale.reshape(N, 1, 2)
        center = bboxes_width_height / 2
        locs = locs / torch.tensor([self.model_input_width, self.model_input_height]) * scale + center - scale / 2

        locs_vals = torch.cat([locs, vals], dim=2)
        return locs_vals

OPSET = 16
RESOLUTIONS = [
    [256,192],
    [384,288],
]

for H, W in RESOLUTIONS:
    get_simcc_maximum_model = \
        get_simcc_maximum_Layer(
            model_input_width=W,
            model_input_height=H,
        )
    simcc_x = torch.randn([1,133,W*2], dtype=torch.float32)
    simcc_y = torch.randn([1,133,H*2], dtype=torch.float32)
    bboxes_width_height = torch.tensor([[218, 346]], dtype=torch.int64)
    MODEL_FILE = f'get_simcc_maximum_1x3x{H}x{W}_{OPSET}.onnx'
    torch.onnx.export(
        get_simcc_maximum_model,
        (simcc_x, simcc_y, bboxes_width_height),
        MODEL_FILE,
        input_names=['post_simcc_x', 'post_simcc_y', 'bboxes_width_height'],
        output_names=['keypoints_xy_score'],
        opset_version=OPSET,
    )
    model_onnx1 = onnx.load(MODEL_FILE)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, MODEL_FILE)
    model_onnx2 = onnx.load(MODEL_FILE)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, MODEL_FILE)
    rename(
        old_new=["/", "post_"],
        input_onnx_file_path=MODEL_FILE,
        output_onnx_file_path=MODEL_FILE,
        mode="full",
        search_mode="prefix_match",
    )

    MODEL_FILE = f'get_simcc_maximum_Nx3x{H}x{W}_{OPSET}.onnx'
    torch.onnx.export(
        get_simcc_maximum_model,
        (simcc_x, simcc_y, bboxes_width_height),
        MODEL_FILE,
        input_names=['post_simcc_x', 'post_simcc_y', 'bboxes_width_height'],
        output_names=['keypoints_xy_score'],
        opset_version=OPSET,
        dynamic_axes={
            'post_simcc_x' : {0: 'batch'},
            'post_simcc_y' : {0: 'batch'},
            'bboxes_width_height' : {0: 'batch'},
            'keypoints_xy_score' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(MODEL_FILE)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, MODEL_FILE)
    model_onnx2 = onnx.load(MODEL_FILE)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, MODEL_FILE)
    rename(
        old_new=["/", "post_"],
        input_onnx_file_path=MODEL_FILE,
        output_onnx_file_path=MODEL_FILE,
        mode="full",
        search_mode="prefix_match",
    )


MODELS = [
    'rtmpose_wholebody_l_1x3x256x192_16',
    'rtmpose_wholebody_l_1x3x384x288_16',
    'rtmpose_wholebody_l_Nx3x256x192_16',
    'rtmpose_wholebody_l_Nx3x384x288_16',
    'rtmpose_wholebody_m_1x3x256x192_16',
    'rtmpose_wholebody_m_Nx3x256x192_16',
    'rtmpose_wholebody_x_1x3x384x288_16',
    'rtmpose_wholebody_x_Nx3x384x288_16',
]
for model_name in MODELS:
    model_name_splits = model_name.split('_')
    combined_graph = combine(
        srcop_destop = [
            ['simcc_x', 'post_simcc_x', 'simcc_y', 'post_simcc_y']
        ],
        input_onnx_file_paths = [
            f'{model_name}.onnx',
            f'get_simcc_maximum_{model_name_splits[3]}_{model_name_splits[4]}.onnx',
        ],
        output_onnx_file_path=f'{model_name}_with_post.onnx',
    )