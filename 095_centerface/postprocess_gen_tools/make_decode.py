import torch
from torch import nn
import onnx
from onnxsim import simplify

class Model(nn.Module):
    def forward(self, heatmap, scale, offset, landmark, threshold=0.5):
        """
        model_input:
            [1, 3, 480, 640]

        model_outputs:
            heatmap = torch.rand((1, 1, 120, 160))
            scale = torch.rand((1, 2, 120, 160))
            offset = torch.rand((1, 2, 120, 160))
            landmark = torch.rand((1, 10, 120, 160))
        """
        batch = heatmap.shape[0]
        input_h = torch.tensor(heatmap.shape[2] * 4)
        input_w = torch.tensor(heatmap.shape[3] * 4)
        output_h = heatmap.shape[2]
        output_w = heatmap.shape[3]

        scale0 = scale[:, 0, :, :] # torch.Size([1, 120, 160])
        scale1 = scale[:, 1, :, :] # torch.Size([1, 120, 160])
        offset0 = offset[:, 0, :, :] # torch.Size([1, 120, 160])
        offset1 = offset[:, 1, :, :] # torch.Size([1, 120, 160])

        """
        heatmapは顔があるピクセルごとの確率値のマトリックス
        heatmap = torch.rand((1, 1, 120, 160))
        _, _, c0, c1 = np.where(heatmap > 0.5)
        c0 = 確率値が閾値を超えたY座標
        c1 = 確率値が閾値を超えたX座標
        独自は閾値で座標をフィルタリングしない
        """
        # c0, c1 = np.where(heatmap > threshold)

        """
        s0とs1はマトリックスにスケールを加味したあとに4倍した値
        o0とo1はheatmapの閾値を超えたポイントにフィルタしたoffsetのリスト
        独自は閾値で座標をフィルタリングしない
        """
        # s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
        # o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
        s0 = torch.unsqueeze(torch.exp(scale0) * 4, dim=3) # torch.Size([1, 120, 160, 1])
        s1 = torch.unsqueeze(torch.exp(scale1) * 4, dim=3) # torch.Size([1, 120, 160, 1])
        o0 = torch.unsqueeze(offset0, dim=3) # torch.Size([1, 120, 160, 1])
        o1 = torch.unsqueeze(offset1, dim=3) # torch.Size([1, 120, 160, 1])

        """
        sはヒートマップマトリックスのうち、確率が閾値を超えた座標のリスト
        独自のbmyxはヒートマップの確率閾値を完全無視して全座標のマトリックスを保持,バッチサイズも考慮
        e.g. torch.Size([3, 120, 160, 2])
            [
                [0,0],[0,1],[0,2], ..., [0,157],[0,158],[0,159],
                [1,0],[1,1],[1,2], ..., [1,157],[1,158],[1,159],
                    :
                [118,0],[118,1],[118,2], ..., [118,157],[118,158],[118,159],
                [119,0],[119,1],[119,2], ..., [119,157],[119,158],[119,159],
            ],
            [
                [0,0],[0,1],[0,2], ..., [0,157],[0,158],[0,159],
                [1,0],[1,1],[1,2], ..., [1,157],[1,158],[1,159],
                    :
                [118,0],[118,1],[118,2], ..., [118,157],[118,158],[118,159],
                [119,0],[119,1],[119,2], ..., [119,157],[119,158],[119,159],
            ],
            [
                [0,0],[0,1],[0,2], ..., [0,157],[0,158],[0,159],
                [1,0],[1,1],[1,2], ..., [1,157],[1,158],[1,159],
                    :
                [118,0],[118,1],[118,2], ..., [118,157],[118,158],[118,159],
                [119,0],[119,1],[119,2], ..., [119,157],[119,158],[119,159],
            ]
        """
        # s = heatmap[c0[i], c1[i]]
        mesh_y_coords = torch.arange(output_h)
        mesh_x_coords = torch.arange(output_w)
        my, mx = torch.meshgrid(mesh_y_coords, mesh_x_coords)
        my = torch.unsqueeze(my, dim=0)
        mx = torch.unsqueeze(mx, dim=0)
        my = torch.unsqueeze(my, dim=3)
        mx = torch.unsqueeze(mx, dim=3)
        myx = torch.cat([my,mx], dim=3)
        bmyx = myx.repeat(batch, 1, 1, 1)

        """
        x1とy1 は座標マトリックスにオフセットを加えてスケールを適用した元画像に対する座標
        座標マトリックス      bmyx  torch.Size([1, 120, 160, 2])
        オフセットマトリックス o0とo1 torch.Size([1, 120, 160, 1])
        スケールマトリックス   s0とs1 storch.Size([1, 120, 160, 1])
        o0はY座標系オフセットのマトリックス
        o1はX座標系オフセットのマトリックス
        s0はY座標系スケールのマトリックス
        s1はX座標系オフセットのマトリックス

        sizeは入力画像の縦横サイズ
        input_h = size[0]
        input_w = size[1]

        y1 torch.Size([1, 120, 160, 1])
        x1 torch.Size([1, 120, 160, 1])
        """
        # x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
        # x1, y1 = min(x1, size[1]), min(y1, size[0])
        y1 = (bmyx[..., 0:1] + o0 + 0.5) * 4 - (s0 / 2)
        x1 = (bmyx[..., 1:2] + o1 + 0.5) * 4 - (s1 / 2)
        y1 = torch.maximum(y1, torch.tensor(0.0))
        x1 = torch.maximum(y1, torch.tensor(0.0))
        y1 = torch.minimum(y1, input_h.to(torch.float32))
        x1 = torch.minimum(x1, input_w.to(torch.float32))

        """
        input_h = size[0]
        input_w = size[1]

        boxesは [x1, y1, x2, y2, score]
            scoreはheatmapそのもの

        y1: torch.Size([1, 120, 160, 1])
        s0: torch.Size([1, 120, 160, 1])
        x1: torch.Size([1, 120, 160, 1])
        s1: torch.Size([1, 120, 160, 1])
        heatmap: torch.rand((1, 1, 120, 160))
        s: torch.Size([1, 120, 160, 1])

        all_boxes: torch.Size([1, 120, 160, 5]) -> torch.Size([1, 19200, 5])
        """
        # boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
        s = heatmap.permute([0,2,3,1])
        y2 = y1 + s0
        x2 = x1 + s1
        y2 = torch.minimum(y2, input_h.to(torch.float32))
        x2 = torch.minimum(x2, input_w.to(torch.float32))
        all_boxes = torch.cat([x1, y1, x2, y2, s], dim=3)
        all_boxes_reshape = all_boxes.reshape([batch, output_h*output_w, 5])

        """
        landmark: torch.rand((1, 10, 120, 160)) -> [1,120,160,10]
            10: y1, x1, y2, x2, y3, x3, y4, x4, y5, x5
        """
        # lm = []
        # for j in range(5):
        #     lm.append(
        #       landmark[
        #           0,
        #           j * 2 + 1,
        #           c0[i],
        #           c1[i],
        #       ] * s1 + x1
        #     )
        #     lm.append(
        #       landmark[
        #           0,
        #           j * 2,
        #           c0[i],
        #           c1[i],
        #       ] * s0 + y1
        #     )
        # lms.append(lm)
        landmark = landmark.permute([0,2,3,1])
        landmark = landmark.reshape([batch, output_h*output_w, 10])
        s0_reshape = s0.reshape([batch, output_h*output_w, 1])
        s1_reshape = s1.reshape([batch, output_h*output_w, 1])
        y1_reshape = y1.reshape([batch, output_h*output_w, 1])
        x1_reshape = x1.reshape([batch, output_h*output_w, 1])
        lm_y1 = landmark[:, :, 0:1] * s0_reshape + y1_reshape
        lm_x1 = landmark[:, :, 1:2] * s1_reshape + x1_reshape
        lm_y2 = landmark[:, :, 2:3] * s0_reshape + y1_reshape
        lm_x2 = landmark[:, :, 3:4] * s1_reshape + x1_reshape
        lm_y3 = landmark[:, :, 4:5] * s0_reshape + y1_reshape
        lm_x3 = landmark[:, :, 5:6] * s1_reshape + x1_reshape
        lm_y4 = landmark[:, :, 6:7] * s0_reshape + y1_reshape
        lm_x4 = landmark[:, :, 7:8] * s1_reshape + x1_reshape
        lm_y5 = landmark[:, :, 8:9] * s0_reshape + y1_reshape
        lm_x5 = landmark[:, :, 9:10] * s1_reshape + x1_reshape
        landmark_yx = torch.cat(
            [
                lm_y1,
                lm_x1,
                lm_y2,
                lm_x2,
                lm_y3,
                lm_x3,
                lm_y4,
                lm_x4,
                lm_y5,
                lm_x5,
            ],
            dim=2
        )

        """
        all_boxes_reshape: torch.Size([1, 19200, 5])
            y1, x1, y2, x2, score
        """
        keep = all_boxes_reshape[:, :, 4:5] >= threshold
        keep = keep.squeeze()
        final_boxes_y1x1y2x2score = all_boxes_reshape[:, keep, :]
        final_landmark_yx = landmark_yx[:, keep, :]


        # boxes = np.asarray(boxes, dtype=np.float32)
        # keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)

        # boxes = boxes[keep, :]
        # lms = np.asarray(lms, dtype=np.float32)
        # lms = lms[keep, :]




        return final_boxes_y1x1y2x2score, final_landmark_yx


model = Model()
model.eval()
H=480
W=640
x1 = torch.rand((1, 1, H//4, W//4))
x2 = torch.rand((1, 2, H//4, W//4))
x3 = torch.rand((1, 2, H//4, W//4))
x4 = torch.rand((1, 10, H//4, W//4))
x5 = torch.rand(1)

onnx_file = 'decode_Nx3xHxW.onnx'
torch.onnx.export(
    model,
    (x1,x2,x3,x4,x5),
    onnx_file,
    opset_version=12,
    input_names=['decode_heatmap', 'decode_scale', 'decode_offset', 'decode_landmark', 'score_threshold'],
    output_names=['decode_boxes_y1x1y2x2score', 'decode_lms_yx'],
    dynamic_axes={
        'decode_heatmap' : {0: 'N', 2: 'height', 3: 'width'},
        'decode_scale' : {0: 'N', 2: 'height', 3: 'width'},
        'decode_offset' : {0: 'N', 2: 'height', 3: 'width'},
        'decode_landmark' : {0: 'N', 2: 'height', 3: 'width'},
        'decode_boxes_y1x1y2x2score' : {0: 'N', 1: 'boxes', 2: '5'},
        'decode_lms_yx' : {0: 'N', 1: 'boxes', 2: '10'},
    }
)
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)


onnx_file = 'decode_1x3xHxW.onnx'
torch.onnx.export(
    model,
    (x1,x2,x3,x4,x5),
    onnx_file,
    opset_version=12,
    input_names=['decode_heatmap', 'decode_scale', 'decode_offset', 'decode_landmark', 'score_threshold'],
    output_names=['decode_boxes_y1x1y2x2score', 'decode_lms_yx'],
    dynamic_axes={
        'decode_heatmap' : {2: 'height', 3: 'width'},
        'decode_scale' : {2: 'height', 3: 'width'},
        'decode_offset' : {2: 'height', 3: 'width'},
        'decode_landmark' : {2: 'height', 3: 'width'},
        'decode_boxes_y1x1y2x2score' : {1: 'boxes'},
        'decode_lms_yx' : {1: 'boxes'},
    }
)
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
