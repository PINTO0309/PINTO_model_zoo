import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
import numpy as np
from functools import partial
from itertools import product, chain
from math import sqrt
from typing import List, Tuple

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from layers.warp_utils import deform_op
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage

import logging
import os

import copy

try:
    from torch2trt import torch2trt
    from torch2trt.torch2trt import TRTModule
    use_torch2trt = True
except:
    use_torch2trt = False

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()
# torch.device('cpu')

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = False if use_torch2trt else torch.cuda.device_count() <= 1
NoneTensor = None if use_torch2trt else torch.Tensor()

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def to_tensorrt(self, int8_mode=False):
        """Converts the bbox, conf, and mask layer of the PredictionModule
           into TRTModules.
        """

        # Each PredictionModule takes a particular input shape.
        # Torch2TRT optimizes based on the input shape so we need to
        # make sure that we feed it the same shape that it will receive
        # during testing phase.
        input_sizes = [
                (1, 256, 69, 69),
                (1, 256, 35, 35),
                (1, 256, 18, 18),
                (1, 256, 9, 9),
                (1, 256, 5, 5),
        ]

        x = torch.ones(input_sizes[self.index]).cuda()

        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        if self.index == 0 and cfg.share_prediction_module:
            self.upfeature_old  = self.upfeature
            self.bbox_layer_old = self.bbox_layer
            self.conf_layer_old = self.conf_layer
            self.mask_layer_old = self.mask_layer

            self.upfeature  = trt_fn(self.upfeature, [x])
            self.bbox_layer = trt_fn(self.bbox_layer, [x])
            self.conf_layer = trt_fn(self.conf_layer, [x])
            self.mask_layer = trt_fn(self.mask_layer, [x])
        elif self.index > 0 and self.parent is not None and cfg.share_prediction_module:
            self.bbox_extra = self.parent[0].bbox_extra
            self.conf_extra = self.parent[0].conf_extra
            self.mask_extra = self.parent[0].mask_extra

            self.upfeature  = trt_fn(self.parent[0].upfeature_old, [x])
            self.bbox_layer = trt_fn(self.parent[0].bbox_layer_old, [x])
            self.conf_layer = trt_fn(self.parent[0].conf_layer_old, [x])
            self.mask_layer = trt_fn(self.parent[0].mask_layer_old, [x])

            self.parent = [None]
        else:
            raise NotImplementedError("to_tensorrt doesn't currently work when we're not"
                                      "sharing the prediction module")

    def forward(self, x):#, extras=None):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        extras = {"backbone": "full", "interrupt": True, "keep_statistics": True, "moving_statistics": {"conf_hist": []}}
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        if extras is not None:
            assert type(extras) == dict
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        
        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if extras is not None and 'conf_hist' in extras:
            conf_hist = extras['conf_hist'].copy()
            _, _, h, w = bbox_x.size()
            conf_spatial = conf.view(x.size(0), h, w, -1).contiguous().permute(0, 3, 1, 2)
            conf_hist.append(conf_spatial)
            stacked_conf = torch.stack(conf_hist, dim=0)
            conf = torch.mean(stacked_conf, dim=0)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)
        
        priors = self.make_priors(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        if cfg.use_instance_coeff:
            preds['inst'] = inst
        
        return preds
    
    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for scale, ars in zip(self.scales, self.aspect_ratios):
                        for ar in ars:
                            if not cfg.backbone.preapply_sqrt:
                                ar = sqrt(ar)

                            if cfg.backbone.use_pixel_scales:
                                if type(cfg.max_size) == tuple:
                                    width, height = cfg.max_size
                                    w = scale * ar / width
                                    h = scale / ar / height
                                else:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h
                            
                            # This is for backward compatability with a bug where I made everything square by accident
                            if cfg.backbone.use_square_anchors:
                                h = w

                            prior_data += [x, y, w, h]
                
                self.priors = torch.Tensor(prior_data).view(-1, 4)
                self.last_conv_size = (conv_w, conv_h)
        
        return self.priors


class PredictionModuleTRT(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

        if cfg.mask_proto_coeff_activation == torch.tanh:
            self.activation_func = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None:
            x = self.upfeature(x)
        
        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = self.block(x)
            
            b = self.conv(x)
            b = self.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = self.bbox_extra(x)
        conf_x = self.conf_extra(x)
        mask_x = self.mask_extra(x)

        bbox = self.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if cfg.eval_mask_branch:
            mask = self.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_instance_coeff:
            inst = self.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)
            raise NotImplementedError

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = self.activation_func(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = self.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        # preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        # if cfg.use_instance_coeff:
        #     preds['inst'] = inst

        return bbox, conf, mask
        
        # return preds
    
    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for scale, ars in zip(self.scales, self.aspect_ratios):
                        for ar in ars:
                            if not cfg.backbone.preapply_sqrt:
                                ar = sqrt(ar)

                            if cfg.backbone.use_pixel_scales:
                                if type(cfg.max_size) == tuple:
                                    width, height = cfg.max_size
                                    w = scale * ar / width
                                    h = scale / ar / height
                                else:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h
                            
                            # This is for backward compatability with a bug where I made everything square by accident
                            if cfg.backbone.use_square_anchors:
                                h = w

                            prior_data += [x, y, w, h]
                
                self.priors = torch.Tensor(prior_data).view(-1, 4)
                self.last_conv_size = (conv_w, conv_h)
        
        return self.priors


def conv_bn_lrelu(in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def deconv_no_relu(in_features, out_features):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
    )


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def deconv(in_features, out_features):
    return nn.Sequential(
        deconv_no_relu(in_features, out_features),
        nn.LeakyReLU(0.1, inplace=True)
    )


def predict_flow(in_features):
    return nn.Conv2d(in_features, 2, kernel_size=3, stride=1, padding=1, bias=False)


def conv_bn_relu(in_features, out_features, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


def conv_relu(in_features, out_features, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation,
                  dilation=dilation, groups=groups),
        nn.ReLU(inplace=True)
    )


def conv_only(in_features, out_features, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation,
            dilation=dilation, groups=groups)


def conv_lrelu(in_features, out_features, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation,
                  dilation=dilation, groups=groups),
        nn.LeakyReLU(0.1, inplace=True)
    )


def conv_bn(in_features, out_features, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation,
                  dilation=dilation),
        nn.BatchNorm2d(out_features)
    )


def shuffle_cat(a, b):
    assert a.size() == b.size()
    n, c, h, w = a.size()
    a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
    b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
    x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
    x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
    return x


class Cat(nn.Module):
    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)
        return x


class ShuffleCat(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
        b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
        x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
        return x


class ShuffleCatChunk(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = torch.chunk(a, chunks=c, dim=1)
        b = torch.chunk(b, chunks=c, dim=1)
        x = [None] * (c * 2)
        x[::2] = a
        x[1::2] = b
        x = torch.cat(x, dim=1)
        return x


class ShuffleCatAlt(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        x = torch.zeros(n, c*2, h, w, dtype=a.dtype, device=a.device)
        x[:, ::2] = a
        x[:, 1::2] = b
        return x


class FlowNetUnwrap(nn.Module):
    def forward(self, preds):
        outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        flow1, scale1, bias1, flow2, scale2, bias2, flow3, scale3, bias3 = preds

        outs.append((flow1, scale1, bias1))
        outs.append((flow2, scale2, bias2))
        outs.append((flow3, scale3, bias3))
        return outs


class FlowNetMiniTRTWrapper(nn.Module):
    def __init__(self, flow_net):
        super().__init__()
        self.flow_net = flow_net
        if cfg.flow.use_shuffle_cat:
            self.cat = ShuffleCat()
        else:
            self.cat = Cat()
        self.unwrap = FlowNetUnwrap()

    def forward(self, a, b):
        concat = self.cat(a, b)
        dummy_tensor = torch.tensor(0, dtype=a.dtype)
        preds = [dummy_tensor, dummy_tensor, dummy_tensor]
        preds_ = self.flow_net(concat)
        preds.extend(preds_)
        outs = self.unwrap(preds)
        return outs


class PredictionModuleTRTWrapper(nn.Module):
    def __init__(self, pred_layer):
        super().__init__()
        self.pred_layer = PredictionModuleTRT(*pred_layer.params[:-2], None, pred_layer.params[-1])

        pred_layer_w = pred_layer.parent[0] if pred_layer.parent[0] is not None else pred_layer
        self.pred_layer.load_state_dict(pred_layer_w.state_dict())

    def to_tensorrt(self, int8_mode=False, calibration_dataset=None):
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        input_sizes = [
                (1, 256, 69, 69),
                (1, 256, 35, 35),
                (1, 256, 18, 18),
                (1, 256, 9, 9),
                (1, 256, 5, 5),
        ]

        x = torch.ones(input_sizes[self.pred_layer.index]).cuda()
        self.pred_layer_torch = self.pred_layer
        self.pred_layer = trt_fn(self.pred_layer, [x])

    def forward(self, x):
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        bbox, conf, mask = self.pred_layer(x)
        priors = self.pred_layer_torch.make_priors(conv_h, conv_w)
        
        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
        
        return preds


class NoReLUBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(NoReLUBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class FlowNetMiniPredLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.conv = nn.Conv2d(in_features, 2, kernel_size=3, padding=1, bias=False)
        self.scale = nn.Conv2d(in_features, cfg.fpn.num_features, kernel_size=1, padding=0, bias=True)
        self.bias = nn.Conv2d(in_features, cfg.fpn.num_features, kernel_size=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        nn.init.constant_(self.scale.bias, 1)

    def forward(self, x):
        offset = self.conv(x)
        scale = self.scale(x)
        bias = self.bias(x)
        return offset, scale, bias


class FlowNetMiniPreConvs(ScriptModuleWrapper):
    def __init__(self, reduce_channels):
        super().__init__()

        last_in_channels = cfg.fpn.num_features
        convs = []
        for reduce_channel in reduce_channels:
            convs.append(conv_lrelu(last_in_channels, reduce_channel))
        
        self.convs = nn.Sequential(*convs)

    @script_method_wrapper
    def forward(self, x):
        return self.convs(x)


def build_flow_convs(encode_layers, in_features, out_features, stride=1, groups=1):
    conv = []
    conv.append(conv_lrelu(in_features, cfg.flow.encode_channels * encode_layers[0], groups=groups, stride=stride))
    for encode_idx, encode_layer in enumerate(encode_layers[:-1]):
        conv.append(conv_lrelu(cfg.flow.encode_channels * encode_layers[encode_idx], cfg.flow.encode_channels * encode_layers[encode_idx + 1], groups=groups))
    conv.append(conv_lrelu(cfg.flow.encode_channels * encode_layers[-1], out_features))
    return nn.Sequential(*conv)


class FlowNetMini(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode', 'use_shuffle_cat', 'skip_flow']
    def __init__(self, in_features):
        super().__init__()
        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.use_shuffle_cat = cfg.flow.use_shuffle_cat

        self.conv1 = build_flow_convs(cfg.flow.encode_layers[0], in_features, cfg.flow.encode_channels, groups=cfg.flow.num_groups)
        self.conv2 = build_flow_convs(cfg.flow.encode_layers[1], cfg.flow.encode_channels, cfg.flow.encode_channels * 2, stride=2)
        self.conv3 = build_flow_convs(cfg.flow.encode_layers[2], cfg.flow.encode_channels * 2, cfg.flow.encode_channels * 4, stride=2)

        self.pred3 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 4)
        self.pred2 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 4 + 2)
        self.pred1 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 2 + 2)

        self.upfeat3 = nn.Conv2d(cfg.flow.encode_channels * 4, cfg.flow.encode_channels * 2,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.upflow3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.upfeat2 = nn.Conv2d(cfg.flow.encode_channels * 4 + 2, cfg.flow.encode_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.upflow2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_flow = not self.training and cfg.flow.flow_layer != 'top' and '3' not in cfg.flow.warp_layers

        for m in chain(*[x.modules() for x in (self.conv1, self.conv2, self.conv3)]):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, 0.1, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    @script_method_wrapper
    def forward(self, target_feat, source_feat):
        preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        if self.use_shuffle_cat:
            concat0 = shuffle_cat(target_feat, source_feat)
        else:
            concat0 = torch.cat((target_feat, source_feat), dim=1)

        out_conv1 = self.conv1(concat0)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)

        _, _, h2, w2 = out_conv2.size()

        flow3, scale3, bias3 = self.pred3(out_conv3)
        out_upfeat3 = F.interpolate(out_conv3, size=(h2, w2), mode=self.interpolation_mode, align_corners=False)
        out_upfeat3 = self.upfeat3(out_upfeat3)
        out_upflow3 = F.interpolate(flow3, size=(h2, w2), mode=self.interpolation_mode, align_corners=False)
        out_upflow3 = self.upflow3(out_upflow3)

        concat2 = torch.cat((out_conv2, out_upfeat3, out_upflow3), dim=1)
        flow2, scale2, bias2 = self.pred2(concat2)

        dummy_tensor = torch.tensor(0, dtype=out_conv2.dtype)

        if not self.skip_flow:
            _, _, h1, w1 = out_conv1.size()
            out_upfeat2 = F.interpolate(concat2, size=(h1, w1), mode=self.interpolation_mode, align_corners=False)
            out_upfeat2 = self.upfeat2(out_upfeat2)
            out_upflow2 = F.interpolate(flow2, size=(h1, w1), mode=self.interpolation_mode, align_corners=False)
            out_upflow2 = self.upflow2(out_upflow2)

            concat1 = torch.cat((out_conv1, out_upfeat2, out_upflow2), dim=1)
            flow1, scale1, bias1 = self.pred1(concat1)

            preds.append((flow1, scale1, bias1))
        else:
            preds.append((dummy_tensor, dummy_tensor, dummy_tensor))
        preds.append((flow2, scale2, bias2))
        preds.append((flow3, scale3, bias3))

        return preds


class FlowNetMiniTRT(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode']
    def __init__(self, in_features):
        super().__init__()
        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.in_features = in_features

        self.conv1 = build_flow_convs(cfg.flow.encode_layers[0], in_features, cfg.flow.encode_channels, groups=cfg.flow.num_groups)
        self.conv2 = build_flow_convs(cfg.flow.encode_layers[1], cfg.flow.encode_channels, cfg.flow.encode_channels * 2, stride=2)
        self.conv3 = build_flow_convs(cfg.flow.encode_layers[2], cfg.flow.encode_channels * 2, cfg.flow.encode_channels * 4, stride=2)

        self.pred3 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 4)
        self.pred2 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 4 + 2)
        self.pred1 = FlowNetMiniPredLayer(cfg.flow.encode_channels * 2 + 2)

        self.upfeat3 = nn.Conv2d(cfg.flow.encode_channels * 4, cfg.flow.encode_channels * 2,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.upflow3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.upfeat2 = nn.Conv2d(cfg.flow.encode_channels * 4 + 2, cfg.flow.encode_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.upflow2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        for m in chain(*[x.modules() for x in (self.conv1, self.conv2, self.conv3)]):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, 0.1, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    @script_method_wrapper
    # def forward(self, target_feat, source_feat):
    def forward(self, concat0):
        # preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        preds = []

        # concat0 = shuffle_cat(target_feat, source_feat)

        out_conv1 = self.conv1(concat0)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)

        _, _, h2, w2 = out_conv2.size()

        flow3, scale3, bias3 = self.pred3(out_conv3)
        out_upfeat3 = F.interpolate(out_conv3, size=(h2, w2), mode=self.interpolation_mode, align_corners=False)
        out_upfeat3 = self.upfeat3(out_upfeat3)
        out_upflow3 = F.interpolate(flow3, size=(h2, w2), mode=self.interpolation_mode, align_corners=False)
        out_upflow3 = self.upflow3(out_upflow3)

        concat2 = torch.cat((out_conv2, out_upfeat3, out_upflow3), dim=1)
        flow2, scale2, bias2 = self.pred2(concat2)

        # out_upfeat2 = F.interpolate(concat2, size=(h1, w1), mode=self.interpolation_mode, align_corners=False)
        # out_upfeat2 = self.upfeat2(out_upfeat2)
        # out_upflow2 = F.interpolate(flow2, size=(h1, w1), mode=self.interpolation_mode, align_corners=False)
        # out_upflow2 = self.upflow2(out_upflow2)

        # concat1 = torch.cat((out_conv1, out_upfeat2, out_upflow2), dim=1)
        # flow1, scale1, bias1 = self.pred1(concat1)

        preds.extend((flow2, scale2, bias2))
        preds.extend((flow3, scale3, bias3))

        return preds


class SPA(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode', 'refine_convs', 'use_normalized_spa']

    def __init__(self, num_layers):
        super().__init__()
        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.use_normalized_spa = cfg.flow.use_normalized_spa

        self.refine_convs = nn.ModuleList([
            conv_lrelu(cfg.fpn.num_features * 2, cfg.fpn.num_features)
            for _ in range(num_layers - 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, 0.1, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    @script_method_wrapper
    def forward(self, c3, f2, f3):
        fpn_outs = [f2, f3]
        out = []

        j = 0

        for refine in self.refine_convs:
            x = fpn_outs[j]
            _, _, h, w = x.size()
            c3 = F.interpolate(c3, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x_normalize = F.normalize(x, dim=1)
            c3_normalize = F.normalize(c3, dim=1)
            if self.use_normalized_spa:
                x = x + refine(torch.cat((x_normalize, c3_normalize), dim=1))
            else:
                x = x + refine(torch.cat((x, c3), dim=1))
            out.append(x)
            j += 1
        return out


class FPN_phase_1(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode', 'lat_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        self.interpolation_mode = cfg.fpn.interpolation_mode

    @script_method_wrapper
    def forward(self, x1=NoneTensor, x2=NoneTensor, x3=NoneTensor, x4=NoneTensor, x5=NoneTensor, x6=NoneTensor, x7=NoneTensor):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        convouts_ = [x1, x2, x3, x4, x5, x6, x7]
        convouts = []
        j = 0
        while j < len(convouts_):
            if convouts_[j] is not None and convouts_[j].size(0):
                convouts.append(convouts_[j])
            j += 1
        # convouts = [x for x in convouts if x is not None]

        out = []
        lat_layers = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)
            lat_layers.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            lat_j = lat_layer(convouts[j])
            lat_layers[j] = lat_j
            x = x + lat_j
            out[j] = x
        
        for i in range(len(convouts)):
            out.append(lat_layers[i])
        return out


class FPN_phase_2(ScriptModuleWrapper):
    __constants__ = ['num_downsample', 'use_conv_downsample', 'pred_layers', 'downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.num_downsample = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample

    @script_method_wrapper
    def forward(self, x1=NoneTensor, x2=NoneTensor, x3=NoneTensor, x4=NoneTensor, x5=NoneTensor, x6=NoneTensor, x7=NoneTensor):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        # out = [x1, x2, x3, x4, x5, x6, x7]
        # out = [x for x in out if x is not None]

        out_ = [x1, x2, x3, x4, x5, x6, x7]
        out = []
        j = 0
        while j < len(out_):
            if out_[j] is not None and out_[j].size(0):
                out.append(out_[j])
            j += 1

        len_convouts = len(out)

        j = len_convouts
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out


class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample',
                     'lat_layers', 'pred_layers', 'downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode  = cfg.fpn.interpolation_mode
        self.num_downsample      = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out


class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, training=True):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        self.training = training

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src
            
            if self.proto_src is None: in_channels = 3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.flow is not None and cfg.flow.proto_net_no_conflict:
                self.proto_net_proxy, _ = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            if cfg.flow is not None:
                self.fpn_phase_1 = FPN_phase_1([src_channels[i] for i in self.selected_layers])
                self.fpn_phase_2 = FPN_phase_2([src_channels[i] for i in self.selected_layers])
                if cfg.flow.fpn_no_conflict:
                    self.fpn_phase_2_proxy = FPN_phase_2([src_channels[i] for i in self.selected_layers])
                if cfg.flow.use_spa or cfg.flow.use_spa_both:
                    self.spa = SPA(len(self.selected_layers))
                if cfg.flow.warp_mode == 'flow':
                    if cfg.flow.model == 'mini':
                        lateral_channels = cfg.fpn.num_features
                        if len(cfg.flow.reduce_channels) > 0:
                            lateral_channels = cfg.flow.reduce_channels[-1]
                        self.flow_net_pre_convs = FlowNetMiniPreConvs(cfg.flow.reduce_channels)
                        # Only use TRT version of FlowNetMini during evaluation and when TensorRT conversion is enforced.
                        if not training and (cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8):
                            self.flow_net = FlowNetMiniTRT(lateral_channels * 2)
                        else:
                            self.flow_net = FlowNetMini(lateral_channels * 2)
                    else:
                        raise NotImplementedError
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            else:
                self.fpn = FPN([src_channels[i] for i in self.selected_layers])
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        if cfg.flow is not None and cfg.flow.pred_heads_no_conflict:
            self.prediction_layers_proxy = nn.ModuleList()

            for idx, layer_idx in enumerate(self.selected_layers):
                # If we're sharing prediction module weights, have every module's parent be the first one
                parent = None
                if cfg.share_prediction_module and idx > 0:
                    parent = self.prediction_layers_proxy[0]

                pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                        aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                        scales        = cfg.backbone.pred_scales[idx],
                                        parent        = parent,
                                        index         = idx)
                self.prediction_layers_proxy.append(pred)

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path, args=None):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path, map_location=torch.device(torch.cuda.current_device()))

        # Get all possible weights
        cur_state_dict = self.state_dict()

        if args is not None and args.drop_weights is not None:
            drop_weight_keys = args.drop_weights.split(',')

        for key in list(state_dict.keys()):
            # For backward compatability, remove these (the new variable is called layers)
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

            if args is not None:
                if args.drop_weights is not None:
                    for drop_key in drop_weight_keys:
                        if key.startswith(drop_key):
                            del state_dict[key]

                if args.coco_transfer:
                    if key.startswith('fpn.lat_layers'):
                        state_dict[key.replace('fpn.', 'fpn_phase_1.')] = state_dict[key]
                        del state_dict[key]
                    elif key.startswith('fpn.') and key in state_dict:
                        state_dict[key.replace('fpn.', 'fpn_phase_2.')] = state_dict[key]
                        del state_dict[key]
                    elif 'COCO' not in cfg.dataset.name and (key.startswith('semantic_seg_conv.') or key.startswith('prediction_layers.0.conf_layer')):
                        del state_dict[key]

        keys_not_exist = []
        keys_not_used = []
        keys_mismatch = []

        for key in list(cur_state_dict.keys()):
            if args is not None:
                if args.drop_weights is not None:
                    for drop_key in drop_weight_keys:
                        if key.startswith(drop_key):
                            state_dict[key] = cur_state_dict[key]

            # for compatibility with models without existing modules
            if key not in state_dict:
                keys_not_exist.append(key)
                state_dict[key] = cur_state_dict[key]
            else:
                # check key size mismatches
                if state_dict[key].size() != cur_state_dict[key].size():
                    keys_mismatch.append(key)
                    state_dict[key] = cur_state_dict[key]


        # for compatibility with models with simpler architectures, remove unused weights.
        for key in list(state_dict.keys()):
            if key not in cur_state_dict:
                keys_not_used.append(key)
                del state_dict[key]

        logger = logging.getLogger("yolact.model.load")
        if len(keys_not_used) > 0:
            logger.warning("Some parameters in the checkpoint are not used: {}".format(", ".join(keys_not_used)))
        if len(keys_not_exist) > 0:
            logger.warning("Some parameters required by the model do not exist in the checkpoint, "
                           "and are initialized as they should be: {}".format(", ".join(keys_not_exist)))
        if len(keys_mismatch) > 0:
            logger.warning("Some parameters in the checkpoint have a different shape in the current model, "
                           "and are initialized as they should be: {}".format(", ".join(keys_mismatch)))

        self.load_state_dict(state_dict)

        if not self.training:
            self.create_partial_backbone()
            if cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8:
                self.create_embed_flow_net()

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path, map_location=torch.device(torch.cuda.current_device()))

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv
            
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                
                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

        if cfg.flow is not None and cfg.flow.fine_tune_layers is not None:
            self.fine_tune_layers()

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def fine_tune_layers(self):
        fine_tune_layers = cfg.flow.fine_tune_layers
        freeze_or_ft = fine_tune_layers[0] == '-'
        if freeze_or_ft:
            fine_tune_layers = fine_tune_layers[1:]
        fine_tune_layer_names = fine_tune_layers.split(',')
        logger = logging.getLogger("yolact.train")
        freeze_layers = []
        fine_tune_layers = []
        for name, module in self.named_children():
            name_in_list = name in fine_tune_layer_names
            if name_in_list == freeze_or_ft:
                freeze_layers.append(name)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            else:
                fine_tune_layers.append(name)
        logger.info("Fine tuning weights of modules: {}".format(", ".join(fine_tune_layers)))
        logger.info("Freezing weights of modules: {}".format(", ".join(freeze_layers)))

    def extra_loss(self, net_outs, gt_net_outs):
        losses = {}

        # feature matching loss
        if cfg.flow.feature_matching_loss is not None:
            def t(fea, layer_idx):
                assert not cfg.flow.fpn_no_conflict
                fpn_net = self.fpn_phase_2
                pred_layer = fpn_net.pred_layers[layer_idx + 1]
                bias = pred_layer.bias.detach() if pred_layer.bias is not None else None
                fea = F.relu(F.conv2d(fea, weight=pred_layer.weight.detach(), bias=bias, stride=pred_layer.stride,
                                      padding=pred_layer.padding))
                return fea

            assert cfg.flow.fm_loss_loc in ("L", "P", "L+P")
            loss_W = 0
            # reverse the direction of outs_phase_1 to make resolution order match flow prediction

            pairs = []

            gt_outs_fpn = gt_net_outs["outs_phase_1"][1:]
            preds_outs_fpn = [net_outs["outs_phase_1"][1:]]
            if net_outs.get("direct_transform", None):
                preds_outs_fpn.append(net_outs["direct_transform"])
            for pred_outs_fpn in preds_outs_fpn:
                for layer_idx in range(2):
                    FPN_GTs = gt_outs_fpn[layer_idx]
                    FPN_preds = pred_outs_fpn[layer_idx]
                    if cfg.flow.fm_loss_loc != "P":
                        pairs.append((FPN_GTs, FPN_preds, ))
                    if cfg.flow.fm_loss_loc != "L":
                        pairs.append((t(FPN_GTs, layer_idx), t(FPN_preds, layer_idx), ))

            for FPN_GTs, FPN_preds in pairs:
                n_, c_ = FPN_preds.size()[:2]
                if cfg.flow.feature_matching_loss == 'SmoothL1':
                    level_loss = F.smooth_l1_loss(FPN_preds, FPN_GTs, reduction="sum")
                    level_loss = level_loss / n_ / c_
                elif cfg.flow.feature_matching_loss == 'cosine':
                    level_loss = F.cosine_similarity(FPN_preds, FPN_GTs)
                    level_loss = (1 - level_loss).mean()
                else:
                    raise NotImplementedError
                loss_W += level_loss

            loss_W /= len(pairs)
            losses['W'] = loss_W * cfg.flow.fm_loss_alpha
        return losses

    @staticmethod
    def visualize_nonlocal(net_outs, prev_images, cur_images):
        from random import random
        import cv2
        from layers.output_utils import undo_image_transformation
        import itertools

        target_size = (64, 64)
        prev_image = F.interpolate(prev_images, target_size, mode='area')[0]
        cur_image = F.interpolate(cur_images, target_size, mode='area')[0]

        prev_image = undo_image_transformation(prev_image, target_size[0], target_size[1]).astype('float32')
        cur_image_ = undo_image_transformation(cur_image, target_size[0], target_size[1]).astype('float32')

        images_stack = []
        image_hw = target_size[0]

        num_grids = 5
        for hh, ww in itertools.product(range(num_grids), range(num_grids)):
            ratio_hh = hh / num_grids
            ratio_ww = ww / num_grids
            rand_hh, rand_ww = int(ratio_hh * image_hw), int(ratio_ww * image_hw)

            cur_image = cur_image_.copy()
            cur_image[rand_hh, rand_ww, :] = (1, 0, 0)
            images_stack.append(prev_image.transpose(2, 0, 1))
            images_stack.append(cur_image.transpose(2, 0, 1))

            for nl_cache in net_outs["nl_caches"]:
                feat_hw = int(nl_cache.size(1) ** 0.5)
                inter_hw = int(nl_cache.size(2) ** 0.5)
                nl_cache = nl_cache.view(-1, feat_hw, feat_hw, inter_hw, inter_hw)

                rand_h = int(feat_hw / image_hw * rand_hh)
                rand_w = int(feat_hw / image_hw * rand_ww)

                heat_map = nl_cache[0, rand_h, rand_w].data.cpu().numpy()

                heat_map_min, heat_map_max = heat_map.min(), heat_map.max()
                heat_map = (heat_map - heat_map_min) / (heat_map_max - heat_map_min)
                heat_map_u8 = (heat_map * 255).astype('uint8')
                heat_map_image = cv2.applyColorMap(heat_map_u8, cv2.COLORMAP_JET)
                heat_map_image = cv2.resize(heat_map_image, target_size)
                heat_map_image = (heat_map_image / 255.0).astype('float32')

                blend_ratio = 0.6
                heatmap = cv2.addWeighted(heat_map_image, blend_ratio, prev_image, 1.0 - blend_ratio, 0)
                heatmap = heatmap.transpose(2, 0, 1)
                images_stack.append(heatmap)
        return images_stack

    def forward_flow(self, extras):
        imgs_1, imgs_2 = extras

        if cfg.flow.model == 'mini':
            feas_1 = self.backbone(imgs_1, partial=True)
            feas_2 = self.backbone(imgs_2, partial=True)

            fea_1 = feas_1[-1].detach()
            fea_2 = feas_2[-1].detach()

            src_lat_layer = self.fpn_phase_1.lat_layers[-1]
            src_lat_1 = src_lat_layer(fea_1).detach()
            src_lat_2 = src_lat_layer(fea_2).detach()

            src_lat_1 = self.flow_net_pre_convs(src_lat_1)
            src_lat_2 = self.flow_net_pre_convs(src_lat_2)

            preds_flow = self.flow_net(src_lat_1, src_lat_2)
            preds_flow = [pred[0] for pred in preds_flow]

        else:
            raise NotImplementedError

        return preds_flow

    def create_embed_flow_net(self):
        if hasattr(self, "flow_net"):
            self.flow_net = FlowNetMiniTRTWrapper(self.flow_net)

    def create_partial_backbone(self):
        if cfg.flow.warp_mode == 'none':
            return

        logger = logging.getLogger("yolact.model.load")
        logger.info("Creating partial backbone...")

        backbone = construct_backbone(cfg.backbone)
        backbone.load_state_dict(self.backbone.state_dict())
        backbone.layers = backbone.layers[:2]

        self.partial_backbone = backbone
        logger.info("Partial backbone created...")
    
    def _get_trt_cache_path(self, module_name, int8_mode=False):
        return "{}.{}{}.trt".format(self.model_path, module_name, ".int8_{}".format(cfg.torch2trt_max_calibration_images) if int8_mode else "")

    def has_trt_cached_module(self, module_name, int8_mode=False):
        module_path = self._get_trt_cache_path(module_name, int8_mode)
        return os.path.isfile(module_path)

    def load_trt_cached_module(self, module_name, int8_mode=False):
        module_path = self._get_trt_cache_path(module_name, int8_mode)
        if not os.path.isfile(module_path):
            return None
        module = TRTModule()
        module.load_state_dict(torch.load(module_path))
        return module

    def save_trt_cached_module(self, module, module_name, int8_mode=False):
        module_path = self._get_trt_cache_path(module_name, int8_mode)
        torch.save(module.state_dict(), module_path)

    def trt_load_if(self, module_name, trt_fn, trt_fn_params, int8_mode=False, parent=None):
        if parent is None: parent=self
        if not hasattr(parent, module_name): return
        module = getattr(parent, module_name)
        trt_cache = self.load_trt_cached_module(module_name, int8_mode)
        if trt_cache is None:
            module = trt_fn(module, trt_fn_params)
            self.save_trt_cached_module(module, module_name, int8_mode)
        else:
            module = trt_cache

        setattr(parent, module_name, module)

    def to_tensorrt_backbone(self, int8_mode=False, calibration_dataset=None):
        """Converts the Backbone to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        x = torch.ones((1, 3, cfg.max_size, cfg.max_size)).cuda()
        # self.backbone = trt_fn(self.backbone, [x])
        # self.partial_backbone = trt_fn(self.partial_backbone, [x])
        self.trt_load_if("backbone", trt_fn, [x], int8_mode)
        self.trt_load_if("partial_backbone", trt_fn, [x], int8_mode)

    def to_tensorrt_protonet(self, int8_mode=False, calibration_dataset=None):
        """Converts ProtoNet to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        x = torch.ones((1, 256, 69, 69)).cuda()
        # self.proto_net = trt_fn(self.proto_net, [x])
        self.trt_load_if("proto_net", trt_fn, [x], int8_mode)

    def to_tensorrt_fpn(self, int8_mode=False, calibration_dataset=None):
        """Converts FPN to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        self.lat_layer = self.fpn_phase_1.lat_layers[-1]

        if cfg.backbone.name == "ResNet50" or cfg.backbone.name == "ResNet101":
            x = [
                torch.randn(1, 512, 69, 69).cuda(),
                torch.randn(1, 1024, 35, 35).cuda(),
                torch.randn(1, 2048, 18, 18).cuda(),
                ]
        elif cfg.backbone.name == "MobileNetV2":
            x = [
                torch.randn(1, 32, 69, 69).cuda(),
                torch.randn(1, 64, 35, 35).cuda(),
                torch.randn(1, 160, 18, 18).cuda(),
                ]
        else:
            raise ValueError("Backbone: {} is not currently supported with TensorRT.".format(cfg.backbone.name))

        self.trt_load_if("fpn_phase_1", trt_fn, x, int8_mode)

        if cfg.backbone.name == "ResNet50" or cfg.backbone.name == "ResNet101":
            x = [
                torch.randn(1, 256, 69, 69).cuda(),
                torch.randn(1, 256, 35, 35).cuda(),
                torch.randn(1, 256, 18, 18).cuda(),
                ]
        elif cfg.backbone.name == "MobileNetV2":
            x = [
                torch.randn(1, 256, 69, 69).cuda(),
                torch.randn(1, 256, 35, 35).cuda(),
                torch.randn(1, 256, 18, 18).cuda(),
                ]
        else:
            raise ValueError("Backbone: {} is not currently supported with TensorRT.".format(cfg.backbone.name))

        self.trt_load_if("fpn_phase_2", trt_fn, x, int8_mode)

        trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        if cfg.backbone.name == "ResNet50" or cfg.backbone.name == "ResNet101":
            x = torch.randn(1, 512, 69, 69).cuda()
        elif cfg.backbone.name == "MobileNetV2":
            x = torch.randn(1, 32, 69, 69).cuda()
        else:
            raise ValueError("Backbone: {} is not currently supported with TensorRT.".format(cfg.backbone.name))

        self.trt_load_if("lat_layer", trt_fn, [x], int8_mode=False)

    def to_tensorrt_prediction_head(self, int8_mode=False, calibration_dataset=None):
        """Converts Prediction Head to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        for idx, pred_layer in enumerate(self.prediction_layers):
            pred_layer = PredictionModuleTRTWrapper(pred_layer)
            pred_layer.to_tensorrt()
            self.prediction_layers[idx] = pred_layer

    def to_tensorrt_spa(self, int8_mode=False, calibration_dataset=None):
        """Converts SPA to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)

        c3 = torch.ones((1, 256, 69, 69)).cuda()
        f2 = torch.ones((1, 256, 35, 35)).cuda()
        f3 = torch.ones((1, 256, 18, 18)).cuda()

        self.trt_load_if("spa", trt_fn, [c3, f2, f3], int8_mode, parent=self.spa)

    def to_tensorrt_flow_net(self, int8_mode=False, calibration_dataset=None):
        """Converts FlowNet to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)


        lateral_channels = cfg.fpn.num_features
        if len(cfg.flow.reduce_channels) > 0:
            lateral_channels = cfg.flow.reduce_channels[-1]
        x = torch.ones((1, lateral_channels * 2, 69, 69)).cuda()
        self.trt_load_if("flow_net", trt_fn, [x], int8_mode, parent=self.flow_net)

    def forward(self, x):#, extras=None):
        """ The input should be of size [batch_size, 3, img_h, img_w] """

        # extras = {"backbone": "full", "interrupt": True, "keep_statistics": False, "moving_statistics": {"conf_hist": []}} # resnet
        extras = {"backbone": "full", "interrupt": True, "keep_statistics": True, "moving_statistics": {"conf_hist": []}} # mobilenetv2

        if cfg.flow.train_flow:
            return self.forward_flow(extras)

        outs_wrapper = {}

        with timer.env('backbone'):
            if cfg.flow is None or extras is None or extras["backbone"] == "full":
                outs = self.backbone(x)

            elif extras is not None and extras["backbone"] == "partial":
                if hasattr(self, 'partial_backbone'):
                    outs = self.partial_backbone(x)
                else:
                    outs = self.backbone(x, partial=True)
            
            else:
                raise NotImplementedError

        if cfg.flow is not None:
            with timer.env('fpn'):
                # extras = {"backbone": "full", "interrupt": False, "keep_statistics": True, "moving_statistics": {"conf_hist": []}}
                # print('@@@@@@@@@@@@@@@@@@ extras:', extras)
                assert type(extras) == dict
                if extras["backbone"] == "full":
                    outs = [outs[i] for i in cfg.backbone.selected_layers]
                    outs_fpn_phase_1_wrapper = self.fpn_phase_1(*outs)
                    outs_phase_1, lats_phase_1 = outs_fpn_phase_1_wrapper[:len(outs)], outs_fpn_phase_1_wrapper[len(outs):]
                    lateral = lats_phase_1[0].detach()
                    moving_statistics = extras["moving_statistics"]

                    if extras.get("keep_statistics", False):
                        outs_wrapper["feats"] = [out.detach() for out in outs_phase_1]
                        outs_wrapper["lateral"] = lateral

                    outs_wrapper["outs_phase_1"] = [out.detach() for out in outs_phase_1]
                else:
                    assert extras["moving_statistics"] is not None
                    moving_statistics = extras["moving_statistics"]
                    outs_phase_1 = moving_statistics["feats"].copy()

                    if cfg.flow.warp_mode != 'take':
                        src_conv = outs[-1].detach()
                        src_lat_layer = self.lat_layer if hasattr(self, 'lat_layer') else self.fpn_phase_1.lat_layers[-1]
                        lateral = src_lat_layer(src_conv).detach()

                    if cfg.flow.warp_mode == "flow":
                        with timer.env('flow'):
                            flows = self.flow_net(self.flow_net_pre_convs(lateral), self.flow_net_pre_convs(moving_statistics["lateral"]))
                            preds_feat = list()
                            if cfg.flow.flow_layer == 'top':
                                flows = [flows[0] for _ in flows]
                            if cfg.flow.warp_layers == 'P4P5':
                                flows = flows[1:]
                                outs_phase_1 = outs_phase_1[1:]
                            for (flow, scale_factor, scale_bias), feat in zip(flows, outs_phase_1):
                                if cfg.flow.flow_layer == 'top':
                                    _, _, h, w = feat.size()
                                    _, _, h_, w_ = flow.size()
                                    if (h, w) != (h_, w_):
                                        flow = F.interpolate(flow, size=(h, w), mode="area")
                                        scale_factor = F.interpolate(scale_factor, size=(h, w), mode="area")
                                        scale_bias = F.interpolate(scale_bias, size=(h, w), mode="area")
                                pred_feat = deform_op(feat, flow)
                                if cfg.flow.use_scale_factor:
                                    pred_feat *= scale_factor
                                if cfg.flow.use_scale_bias:
                                    pred_feat += scale_bias
                                preds_feat.append(pred_feat)
                            outs_wrapper["preds_flow"] = [[x.detach() for x in flow] for flow in flows]
                        outs_phase_1 = preds_feat

                    if cfg.flow.warp_layers == 'P4P5':
                        with timer.env('p3'):
                            _, _, h, w = src_conv.size()
                            src_fpn = outs_phase_1[0]
                            src_fpn = F.interpolate(src_fpn, size=(h, w), mode=cfg.fpn.interpolation_mode, align_corners=False)
                            p3 = src_fpn + lateral

                            outs_phase_1 = [p3] + outs_phase_1

                    if cfg.flow.use_spa:
                        with timer.env('spa'):
                            fpn_outs = outs_phase_1.copy()
                            outs_phase_1 = [fpn_outs[0]]
                            outs_ = self.spa(lateral, *fpn_outs[1:])
                            outs_phase_1.extend(outs_)

                    outs_wrapper["outs_phase_1"] = outs_phase_1.copy()

                fpn_phase_2 = self.fpn_phase_2_proxy \
                    if cfg.flow is not None and cfg.flow.fpn_no_conflict and extras["backbone"] == "partial" \
                    else self.fpn_phase_2

                outs = fpn_phase_2(*outs_phase_1)
                if extras["backbone"] == "partial":
                    outs_wrapper["outs_phase_2"] = [out for out in outs]
                else:
                    outs_wrapper["outs_phase_2"] = [out.detach() for out in outs]
        elif cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)

        if extras is not None and extras.get("interrupt", None):
            return outs_wrapper

        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                if cfg.flow is not None and cfg.flow.proto_net_no_conflict and extras["backbone"] == "partial":
                    proto_out = self.proto_net_proxy(proto_x)
                else:
                    proto_out = self.proto_net(proto_x)

                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                if cfg.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if cfg.use_instance_coeff:
                pred_outs['inst'] = []

            prediction_layers = self.prediction_layers_proxy \
                if cfg.flow is not None and cfg.flow.pred_heads_no_conflict and extras["backbone"] == "partial" \
                else self.prediction_layers
            
            for idx, pred_layer in zip(self.selected_layers, prediction_layers):
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                # This is re-enabled during training or non-TRT inference.
                if self.training or not (cfg.torch2trt_prediction_module or cfg.torch2trt_prediction_module_int8):
                    # A hack for the way dataparallel works
                    if cfg.share_prediction_module and pred_layer is not prediction_layers[0]:
                        pred_layer.parent = [prediction_layers[0]]

                p = pred_layer(pred_x)
                
                for k, v in p.items():
                    pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:
            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            outs_wrapper["pred_outs"] = pred_outs
        else:
            if cfg.use_sigmoid_focal_loss:
                # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
            elif cfg.use_objectness_score:
                # See focal_loss_sigmoid in multibox_loss.py for details
                objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                pred_outs['conf'][:, :, 0 ] = 1 - objectness
            else:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            extras = {}
            outs_wrapper["pred_outs"] = self.detect(pred_outs, extras=extras)
            # print('@@@@@@@@@@@@@@', outs_wrapper)
        return outs_wrapper


# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    net = net.cuda()
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
