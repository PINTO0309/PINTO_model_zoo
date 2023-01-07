import torch
import torch.nn as nn
from sor4onnx import rename
from snc4onnx import combine
from soa4onnx import outputs_add
from sod4onnx import outputs_delete


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.score_th = 0.2

    def forward(self, keypoints, descriptors, scores, prev_keypoints, prev_descriptors):
        indicies = scores[:, 0] > self.score_th
        keypoints = keypoints[indicies, :]
        descriptors = descriptors[indicies, :]
        sim = prev_descriptors @ descriptors.permute(1,0)
        sim[sim < 0.9] = 0
        nn12 = torch.argmax(sim, axis=1)
        nn21 = torch.argmax(sim, axis=0)
        ids1 = torch.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]]).permute(1,0)
        matched_keypoints1 = prev_keypoints[matches[:, 0]]
        matched_keypoints2 = keypoints[matches[:, 1]]
        return matched_keypoints1, matched_keypoints2

model = Model()


DESCRIPTION_SIZES = {
    "t": 64,
    "s": 96,
    "n": 128,
    "l": 128,
}
OPSETS=[11,16]

for type, size in DESCRIPTION_SIZES.items():
    keypoints = torch.randn([5000, 2])
    descriptors = torch.randn([5000, size])
    scores = torch.randn([5000, 1])
    prev_keypoints = torch.randn([5000, 2])
    prev_descriptors = torch.randn([5000, size])
    for opset in OPSETS:
        onnx_file = f'alike_mnn_matcher_{type}_{opset}.onnx'
        torch.onnx.export(
            model,
            args=(keypoints,descriptors,scores,prev_keypoints,prev_descriptors),
            f=onnx_file,
            opset_version=opset,
            input_names=[
                'post_keypoints',
                'post_descriptors',
                'post_scores',
                'prev_keypoints',
                'prev_descriptors',
            ],
            output_names=[
                'matched_keypoints1_xy',
                'matched_keypoints2_xy',
            ],
            dynamic_axes={
                'prev_keypoints': {0: 'M'},
                'prev_descriptors': {0: 'M'},
                'matched_keypoints1_xy': {0: 'N'},
                'matched_keypoints2_xy': {0: 'N'},
            }
        )
        import onnx
        from onnxsim import simplify
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        rename(
            old_new=['/Greater','score_threshold_judgment'],
            input_onnx_file_path=onnx_file,
            output_onnx_file_path=onnx_file,
            non_verbose=True,
        )
        rename(
            old_new=['/Constant_2_output_0','score_threshold'],
            input_onnx_file_path=onnx_file,
            output_onnx_file_path=onnx_file,
            non_verbose=True,
        )
        rename(
            old_new=['/','post_'],
            search_mode='prefix_match',
            input_onnx_file_path=onnx_file,
            output_onnx_file_path=onnx_file,
            non_verbose=True,
        )

RESOLUTIONS = [
    [192,320],
    [192,416],
    [192,640],
    [192,800],
    [256,320],
    [256,416],
    [256,640],
    [256,800],
    [256,960],
    [288,480],
    [288,640],
    [288,800],
    [288,960],
    [288,1280],
    [384,480],
    [384,640],
    [384,800],
    [384,960],
    [384,1280],
    [480,640],
    [480,800],
    [480,960],
    [480,1280],
    [544,800],
    [544,960],
    [544,1280],
    [736,1280],
]
for type, size in DESCRIPTION_SIZES.items():
    for opset in OPSETS:
        for h, w in RESOLUTIONS:
            input_model_file = f'alike_{type}_opset{opset}_{h}x{w}.onnx'
            post_process_file = f'alike_mnn_matcher_{type}_{opset}.onnx'
            output_model_file = f'alike_{type}_opset{opset}_{h}x{w}_post.onnx'
            combine(
                srcop_destop=[
                    [
                        'keypoints','post_keypoints',
                        'descriptors','post_descriptors',
                        'scores','post_scores',
                    ],
                ],
                input_onnx_file_paths=[
                    input_model_file,
                    post_process_file
                ],
                output_onnx_file_path=output_model_file,
                non_verbose=True,
            )
            outputs_add(
                input_onnx_file_path=output_model_file,
                output_op_names=[
                    'keypoints',
                    'descriptors',
                ],
                output_onnx_file_path=output_model_file,
                non_verbose=True,
            )
            outputs_delete(
                input_onnx_file_path=output_model_file,
                output_op_names=[
                    'scores_map',
                ],
                output_onnx_file_path=output_model_file,
                non_verbose=True,
            )
