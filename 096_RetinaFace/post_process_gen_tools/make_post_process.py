import argparse
import torch
import torch.nn as nn
from math import ceil
from itertools import product as product
import onnx
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine

cfg_mnet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

class PriorBox(nn.Module):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.name = "s"

    def forward(self, image_height, image_width):
        self.image_size = (image_height, image_width)
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()

    def forward(self, image_height, image_width):
        scale = torch.as_tensor(
            [image_width, image_height, image_width, image_height],
            dtype=torch.float32,
        )
        return scale


class Scale1(nn.Module):
    def __init__(self):
        super(Scale1, self).__init__()

    def forward(self, image_height, image_width):
        scale1 = torch.as_tensor(
            [
                image_width,
                image_height,
                image_width,
                image_height,
                image_width,
                image_height,
                image_width,
                image_height,
                image_width,
                image_height,
            ],
            dtype=torch.float32,
        )
        return scale1


class PrenNMS(nn.Module):
    def __init__(self, cfg):
        super(PrenNMS, self).__init__()

        self.cfg = cfg

    def decode(self, loc, priors, variances):
        boxes = torch.cat(
            (
                priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
                priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1]),
            ),
            dim=2,
        )
        # boxes[..., :2] -= boxes[..., 2:] / 2
        # boxes[..., 2:] += boxes[..., :2]
        # return boxes
        x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
        x2y2 = boxes[..., :2] + boxes[..., 2:] / 2
        x1y1x2y2 = torch.cat([x1y1, x2y2], dim=2)
        return x1y1x2y2

    def decode_landm(self, pre, priors, variances):
        landms = torch.cat(
            (
                priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
                priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
                priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
                priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
                priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
            ),
            dim=2,
        )
        return landms

    def forward(
        self,
        loc,
        conf,
        landms,
        prior_data,
        scale,
        scale1,
    ):
        # name: loc
        #   type: float32 [1,12600,4]
        # name: conf
        #   type: float32 [1,12600,2]
        # name: landms
        #   type: float32 [1,12600,10]
        # name: prior_data (anchor)
        #   type: float32 [12600,4]
        # name: scale
        #   type: float32 [4]
        # name: scale1
        #   type: float32 [10]

        boxes = self.decode(loc, prior_data, self.cfg["variance"])
        boxes = boxes * scale
        boxes_xyxy = boxes
        boxes_yxyx = torch.cat(
            [
                boxes[..., 1:2],
                boxes[..., 0:1],
                boxes[..., 3:4],
                boxes[..., 2:3],
            ],
            dim=2,
        )

        scores = conf[..., 1:2].permute(0, 2, 1)
        landms_copy = self.decode_landm(landms, prior_data, self.cfg["variance"])
        landms_copy = landms_copy * scale1
        return boxes_xyxy, boxes_yxyx, scores, landms_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ih',
        '--image_height',
        type=int,
        default=480,
    )
    parser.add_argument(
        '-iw',
        '--image_width',
        type=int,
        default=640,
    )
    parser.add_argument(
        '-os',
        '--opset',
        type=int,
        default=11,
    )
    args = parser.parse_args()

    # anchor ##############################################################
    priorbox = PriorBox(cfg=cfg_mnet)
    model_file = f'21_post_process_anchors_{args.image_height}x{args.image_width}_{args.opset}.onnx'
    torch.onnx.export(
        priorbox,
        (args.image_height, args.image_width),
        model_file,
        input_names=['image_height', 'image_width'],
        output_names=['anchors'],
        opset_version=args.opset,
    )
    model_onnx1 = onnx.load(model_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    rename(
        old_new=["Constant_0", "constant_anchors"],
        input_onnx_file_path=model_file,
        output_onnx_file_path=model_file,
        mode="full",
        search_mode="prefix_match",
    )

    anchor_box_num = model_onnx1.graph.output[0].type.tensor_type.shape.dim[0].dim_value


    # scale ##############################################################
    scale = Scale()
    model_file = f'22_post_process_scale_{args.image_height}x{args.image_width}_{args.opset}.onnx'
    torch.onnx.export(
        scale,
        (args.image_height, args.image_width),
        model_file,
        input_names=['image_height', 'image_width'],
        output_names=['scale'],
        opset_version=args.opset,
    )
    model_onnx1 = onnx.load(model_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    rename(
        old_new=["Constant_0", "constant_scale"],
        input_onnx_file_path=model_file,
        output_onnx_file_path=model_file,
        mode="full",
        search_mode="prefix_match",
    )

    # scale1 ##############################################################
    scale = Scale1()
    model_file = f'23_post_process_scale1_{args.image_height}x{args.image_width}_{args.opset}.onnx'
    torch.onnx.export(
        scale,
        (args.image_height, args.image_width),
        model_file,
        input_names=['image_height', 'image_width'],
        output_names=['scale1'],
        opset_version=args.opset,
    )
    model_onnx1 = onnx.load(model_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    rename(
        old_new=["Constant_0", "constant_scale1"],
        input_onnx_file_path=model_file,
        output_onnx_file_path=model_file,
        mode="full",
        search_mode="prefix_match",
    )

    # pre-NMS ##############################################################
    prenms = PrenNMS(cfg=cfg_mnet)
    model_file = f'24_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx'
    # name: loc
    #   type: float32 [1,12600,4]
    # name: conf
    #   type: float32 [1,12600,2]
    # name: landms
    #   type: float32 [1,12600,10]
    # name: prenms_scale
    #   type: float32 [4]
    # name: prenms_scale1
    #   type: float32 [10]
    prenms_loc = torch.randn([1, anchor_box_num, 4])
    prenms_conf = torch.randn([1, anchor_box_num, 2])
    prenms_landms = torch.randn([1, anchor_box_num, 10])
    prenms_prior_data = torch.randn([anchor_box_num, 4])
    prenms_scale = torch.randn([4])
    prenms_scale1 = torch.randn([10])
    torch.onnx.export(
        prenms,
        (
            prenms_loc,
            prenms_conf,
            prenms_landms,
            prenms_prior_data,
            prenms_scale,
            prenms_scale1,
        ),
        model_file,
        input_names=[
            'prenms_loc',
            'prenms_conf',
            'prenms_landms',
            'prenms_prior_data',
            'prenms_scale',
            'prenms_scale1',
        ],
        output_names=[
            'prenms_output_boxes_xyxy',
            'prenms_output_boxes_yxyx',
            'prenms_output_scores',
            'prenms_output_landms',
        ],
        opset_version=args.opset,
    )
    model_onnx1 = onnx.load(model_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, model_file)
    model_onnx2 = onnx.load(model_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, model_file)
    model_onnx2 = onnx.load(model_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, model_file)
    model_onnx2 = onnx.load(model_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, model_file)


    # pre-NMS-merge ##############################################################
    combine(
        srcop_destop = [
            ['anchors', 'prenms_prior_data']
        ],
        input_onnx_file_paths = [
            f'21_post_process_anchors_{args.image_height}x{args.image_width}_{args.opset}.onnx',
            f'24_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx'
        ],
        output_onnx_file_path = f'25_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx',
        non_verbose = True,
    )
    combine(
        srcop_destop = [
            ['scale', 'prenms_scale']
        ],
        input_onnx_file_paths = [
            f'22_post_process_scale_{args.image_height}x{args.image_width}_{args.opset}.onnx',
            f'25_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx'
        ],
        output_onnx_file_path = f'26_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx',
        non_verbose = True,
    )
    combine(
        srcop_destop = [
            ['scale1', 'prenms_scale1']
        ],
        input_onnx_file_paths = [
            f'23_post_process_scale1_{args.image_height}x{args.image_width}_{args.opset}.onnx',
            f'26_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx'
        ],
        output_onnx_file_path = f'27_post_process_prenms_1x{args.image_height}x{args.image_width}_{args.opset}.onnx',
        non_verbose = True,
    )








if __name__ == '__main__':
    main()