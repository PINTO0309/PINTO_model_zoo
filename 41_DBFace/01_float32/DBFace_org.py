
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


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
            HSigmoid()
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


class CBNModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBNModule, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=2, stride=2,  bias=False):
        super(UpModule, self).__init__()
        self.dconv = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()
    
    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ContextModule(nn.Module):
    def __init__(self, inchannel):
        super(ContextModule, self).__init__()
    
        self.inconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)

        half = inchannel // 2
        self.upconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv2 = CBNModule(half, half, 3, 1, padding=1)

    def forward(self, x):

        x = self.inconv(x)
        up, down = torch.chunk(x, 2, dim=1)
        up = self.upconv(up)
        down = self.downconv(down)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):
    def __init__(self, inchannel):
        super(DetectModule, self).__init__()
    
        self.upconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)
        self.context = ContextModule(inchannel)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


class DBFace(nn.Module):
    def __init__(self):
        super(DBFace, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),           # 0
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),           # 1
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),           # 2
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),   # 3
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(3, 40, 240, 80, HSwish(), None, 2),                       # 6
            Block(3, 80, 200, 80, HSwish(), None, 1),                       # 7
            Block(3, 80, 184, 80, HSwish(), None, 1),                       # 8
            Block(3, 80, 184, 80, HSwish(), None, 1),                       # 9
            Block(3, 80, 480, 112, HSwish(), SeModule(112), 1),             # 10
            Block(3, 112, 672, 112, HSwish(), SeModule(112), 1),            # 11
            Block(5, 112, 672, 160, HSwish(), SeModule(160), 1),            # 12
            Block(5, 160, 672, 160, HSwish(), SeModule(160), 2),            # 13
            Block(5, 160, 960, 160, HSwish(), SeModule(160), 1),            # 14
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = HSwish()

        self.conv3 = CBNModule(960, 320, kernel_size=1, stride=1, padding=0, bias=False) # 32
        self.conv4 = CBNModule(320, 24, kernel_size=1, stride=1, padding=0, bias=False) # 32
        self.conn0 = CBNModule(24, 24, 1, 1)  # s4
        self.conn1 = CBNModule(40, 24, 1, 1)  # s8
        self.conn3 = CBNModule(160, 24, 1, 1)  # s16

        self.up0 = UpModule(24, 24, 2, 2) # s16
        self.up1 = UpModule(24, 24, 2, 2) # s8
        self.up2 = UpModule(24, 24, 2, 2) # s4
        self.cout = DetectModule(24)
        self.head_hm = nn.Conv2d(48, 1, 1)
        self.head_tlrb = nn.Conv2d(48, 1 * 4, 1)
        self.head_landmark = nn.Conv2d(48, 1 * 10, 1)


    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))

        keep = {"2": None, "5": None, "12": None}
        for index, item in enumerate(self.bneck):
            out = item(out)

            if str(index) in keep:
                keep[str(index)] = out

        out = self.hs2(self.bn2(self.conv2(out)))
        s32 = self.conv3(out)
        s32 = self.conv4(s32)
        s16 = self.up0(s32) + self.conn3(keep["12"])
        s8 = self.up1(s16) + self.conn1(keep["5"])
        s4 = self.up2(s8) + self.conn0(keep["2"])
        out = self.cout(s4)

        hm = self.head_hm(out)
        tlrb = self.head_tlrb(out)
        landmark = self.head_landmark(out)

        sigmoid_hm = hm.sigmoid()
        tlrb = torch.exp(tlrb)
        return sigmoid_hm, tlrb, landmark


    def load(self, file):
        print(f"load model: {file}")

        if torch.cuda.is_available():
            checkpoint = torch.load(file)
        else:
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint)


    