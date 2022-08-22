from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   define make_last_layers, include 7 conv.
#   first five for feature extraction, latter two for prediction.
#------------------------------------------------------------------------#
def make_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

class RIEBody(nn.Module):
    def __init__(self, config):
        super(RIEBody, self).__init__()
        self.config = config
        self.backbone = darknet53(None) #   13,13,1024

        out_filters = self.backbone.layers_out_filters           # out_filters : [64, 128, 256, 512, 1024]

        #------------------------------------------------------------------------#
        #   set two Output Projectors
        #------------------------------------------------------------------------#
        final_out_filter0 = len(config["RIE"]["anchors"][0]) * (5 + config["RIE"]["classes"])   # final_out_filter0 = 75
        self.OPr = make_layers([512, 1024], out_filters[-1], final_out_filter0)
        self.OPs = make_layers([512, 1024], out_filters[-1], final_out_filter0)



    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        x0 = self.backbone(x) # 13,13,1024
        print(x0.shape)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0, out0_branch = _branch(self.OPr, x0)

        out1, out1_branch = _branch(self.OPs, x0)


        return out0

