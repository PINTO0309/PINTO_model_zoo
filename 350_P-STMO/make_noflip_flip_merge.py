import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return (x + y) / 2

model = Model()

BATCH=1
x = torch.randn([BATCH,17,3])
onnx_file = f'noflip_flip_merge_{BATCH}.onnx'
torch.onnx.export(
    model,
    args=(x,x),
    f=onnx_file,
    opset_version=11,
    input_names=[
        'input_noflip',
        'input_flip',
    ],
    output_names=[
        'batch_joints_xyz',
    ],
)
import onnx
from onnxsim import simplify
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)

from sor4onnx import rename
rename(
    old_new=['/','merge_'],
    input_onnx_file_path=onnx_file,
    output_onnx_file_path=onnx_file,
    search_mode='prefix_match',
)

"""
snc4onnx \
--input_onnx_file_paths pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_1x2x10x17_noflip.onnx noflip_flip_merge_1.onnx \
--output_onnx_file_path merged.onnx \
--srcop_destop noflip_batch_joints_xyz input_noflip

snc4onnx \
--input_onnx_file_paths pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_1x2x10x17_flip.onnx merged.onnx \
--output_onnx_file_path pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_1x2x10x17_noflip_flip.onnx \
--srcop_destop flip_batch_joints_xyz input_flip

rm merged.onnx
"""

from snc4onnx import combine

FRAMES = [
    10,
    15,
    20,
    30,
    60,
    100,
    200,
    243,
]

for FRAME in FRAMES:
    combine(
        srcop_destop=[
            ['noflip_batch_joints_xyz', 'input_noflip'],
        ],
        input_onnx_file_paths = [
            f'pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_{BATCH}x2x{FRAME}x17_noflip.onnx',
            f'noflip_flip_merge_{BATCH}.onnx',
        ],
        output_onnx_file_path = 'merged.onnx',
    )
    combine(
        srcop_destop=[
            ['flip_batch_joints_xyz', 'input_flip'],
        ],
        input_onnx_file_paths = [
            f'pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_{BATCH}x2x{FRAME}x17_flip.onnx',
            'merged.onnx',
        ],
        output_onnx_file_path = f'pstmos_in_the_wild_BATCHxXYxFRAMESxJOINTS_{BATCH}x2x{FRAME}x17_noflip_flip.onnx',
    )
