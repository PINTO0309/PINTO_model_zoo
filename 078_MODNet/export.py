import torch
import torch.nn as nn
from src.models.modnet import MODNet
from torch.autograd import Variable

modnet = MODNet(backbone_pretrained=False)
# modnet = nn.DataParallel(modnet).cuda()
# modnet.load_state_dict(torch.load('pretrained/modnet_webcam_portrait_matting.ckpt'))
# modnet.eval()
# torch.save(modnet.module.state_dict(), 'modnet_512x672_float32.pth')

modnet.load_state_dict(torch.load('modnet_512x672_float32.pth'))
modnet.eval()
dummy_input = Variable(torch.randn(1, 3, 512, 512))
torch.onnx.export(modnet, dummy_input, 'modnet_512x512_float32.onnx', export_params=True, opset_version=12)