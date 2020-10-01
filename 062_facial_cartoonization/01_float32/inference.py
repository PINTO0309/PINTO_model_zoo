import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.nn import functional as F




class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output



class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel*2)
        self.down3 = DownBlock(num_channel*2, num_channel*3)
        self.down4 = DownBlock(num_channel*3, num_channel*4)
        res_blocks = [ResBlock(num_channel*4)]*num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel*4, num_channel*3)
        self.up2 = UpBlock(num_channel*3, num_channel*2)
        self.up3 = UpBlock(num_channel*2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1+down3)
        up3 = self.up3(up2+down2)
        up4 = self.up4(up3+down1)
        return up4



if __name__ == '__main__':
    weight = torch.load('weight.pth', map_location='cpu')
    model = SimpleGenerator()
    model.load_state_dict(weight)
    #torch.save(model.state_dict(), 'weight.pth')
    model.eval()
    
    name_list = os.listdir('images')
    name_list = [f for f in name_list if '.jpg' in f]
    if not os.path.exists('results'):
        os.mkdir('results')
    for name in tqdm(name_list):
        load_path = os.path.join('images', name)
        save_path = os.path.join('results', name)
        raw_image = cv2.imread(load_path)
        image = raw_image/127.5 - 1
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).unsqueeze(0)
        output = model(image.float())
        output = output.squeeze(0).detach().numpy()
        output = output.transpose(1, 2, 0)
        output = (output + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = np.concatenate([raw_image, output], axis=1)
        cv2.imwrite(save_path, output)
        
        
            

        
        
    