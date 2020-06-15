
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import numpy as np

_MODEL_URL_DOMAIN = "http://zifuture.com:1000/fs/public_models"
_MODEL_URL_LARGE = "mbv3large-76f5a50e.pth"
_MODEL_URL_SMALL = "mbv3small-09ace125.pth"

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(self.pool(x))


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class Mbv3SmallFast(nn.Module):
    def __init__(self):
        super(Mbv3SmallFast, self).__init__()

        self.keep = [0, 2, 7]
        self.uplayer_shape = [16, 24, 48]
        self.output_channels = 96

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 2),       # 0 *
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),               # 1
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),               # 2 *
            Block(5, 24, 96, 40, nn.ReLU(inplace=True), SeModule(40), 2),                    # 3
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),                   # 4
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),                   # 5
            Block(5, 40, 120, 48, nn.ReLU(inplace=True), SeModule(48), 1),                   # 6
            Block(5, 48, 144, 48, nn.ReLU(inplace=True), SeModule(48), 1),                   # 7 *
            Block(5, 48, 288, 96, nn.ReLU(inplace=True), SeModule(96), 2),                   # 8
        )


    def load_pretrain(self):
        checkpoint = model_zoo.load_url(f"{_MODEL_URL_DOMAIN}/{_MODEL_URL_SMALL}")
        self.load_state_dict(checkpoint, strict=False)

        
    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))

        outs = []
        for index, item in enumerate(self.bneck):
            x = item(x)

            if index in self.keep:
                outs.append(x)

        outs.append(x)
        return outs


# Conv BatchNorm Activation
class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Up Sample Module
class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,  bias=False, mode="UCBA"):
        super(UpModule, self).__init__()
        self.mode = mode

        if self.mode == "UCBA":
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias)
        elif self.mode == "DeconvBN":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.mode == "DeCBA":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError(f"Unsupport mode: {mode}")
    
    def forward(self, x):
        if self.mode == "UCBA":
            return self.conv(self.up(x))
        elif self.mode == "DeconvBN":
            return F.relu(self.bn(self.dconv(x)))
        elif self.mode == "DeCBA":
            return self.conv(self.dconv(x))


# SSH Context Module
class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()

        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1)

    def forward(self, x):

        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


# SSH Detect Module
class DetectModule(nn.Module):
    def __init__(self, in_channels):
        super(DetectModule, self).__init__()

        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1)
        self.context = ContextModule(in_channels)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


# Job Head Module
class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, has_ext=False):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.has_ext = has_ext

        if has_ext:
            self.ext = CBAModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):

        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


# DBFace Model
class DBFace(nn.Module):
    def __init__(self, has_landmark=True, wide=64, has_ext=True, upmode="UCBA"):
        super(DBFace, self).__init__()
        self.has_landmark = has_landmark

        # define backbone
        self.bb = Mbv3SmallFast()

        # Get the number of branch node channels
        # stride4, stride8, stride16
        c0, c1, c2 = self.bb.uplayer_shape

        self.conv3 = CBAModule(self.bb.output_channels, wide, kernel_size=1, stride=1, padding=0, bias=False) # s32
        self.connect0 = CBAModule(c0, wide, kernel_size=1)  # s4
        self.connect1 = CBAModule(c1, wide, kernel_size=1)  # s8
        self.connect2 = CBAModule(c2, wide, kernel_size=1)  # s16

        self.up0 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode) # s16
        self.up1 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode) # s8
        self.up2 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode) # s4
        self.detect = DetectModule(wide)

        self.center = HeadModule(wide, 1, has_ext=has_ext)
        self.box = HeadModule(wide, 4, has_ext=has_ext)

        if self.has_landmark:
            self.landmark = HeadModule(wide, 10, has_ext=has_ext)


    def init_weights(self):

        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19

        # Load backbone weights from ImageNet
        self.bb.load_pretrain()
        self.center.init_normal(0.001, d)
        self.box.init_normal(0.001, 0)

        if self.has_landmark:
            self.landmark.init_normal(0.001, 0)


    def load(self, file):
        checkpoint = torch.load(file, map_location="cpu")
        self.load_state_dict(checkpoint)


    def forward(self, x):
        s4, s8, s16, s32 = self.bb(x)
        s32 = self.conv3(s32)

        s16 = self.up0(s32) + self.connect2(s16)
        s8 = self.up1(s16) + self.connect1(s8)
        s4 = self.up2(s8) + self.connect0(s4)
        x = self.detect(s4)

        center = self.center(x)
        box = self.box(x)

        center = center.sigmoid()
        box = torch.exp(box)

        if self.has_landmark:
            landmark = self.landmark(x)
            return center, box, landmark

        return center, box

model = DBFace().cuda()
model.eval()
model.load("dbfaceSmallH.pth")

dummy_input = torch.zeros((1, 3, 256, 256)).cuda()
torch.onnx.export(model, dummy_input, "dbfaceSmallH.onnx", output_names=["center", "box", "landmark"])
