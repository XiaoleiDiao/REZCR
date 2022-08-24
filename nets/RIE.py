from collections import OrderedDict
import math
import torch
import torch.nn as nn
from utils.config import Config
from nets.DSAL import DSAL_layer

class CBR(nn.Module):
    def __init__(self, inplanes, planes):
        super(CBR, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual

        return out



class RAB(nn.Module):
    def __init__(self, layers, kernel_size = 7):
        super(RAB, self).__init__()
        self.inplanes = [32,64,128,256,512]
        self.planes = [3, 32, 64,128,256,512,1024,1024]
        # print(self.planes[1:4])

        self.conv1 = nn.Conv2d(3, self.inplanes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes[0])
        self.relu1 = nn.LeakyReLU(0.1)


        self.layer1 = self._make_layer(self.planes[1:3], layers[0])
        self.layer2 = self._make_layer(self.planes[2:4], layers[1])
        self.layer3 = self._make_layer(self.planes[3:5], layers[2])
        self.layer4 = self._make_layer(self.planes[4:6], layers[3])
        self.layer5 = self._make_layer(self.planes[5:7], layers[4])
        self.DSAL = DSAL_layer(kernel_size)


        self.layers_out_filters = [32, 64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(planes[0], planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        for i in range(0, blocks):
            layers.append(("CBR_{}".format(i), CBR(planes[1], planes)))
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.DSAL(x)

        return out


def RABn(pretrained, **kwargs):
    model = RAB([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   define last layers for two output projectors.
#------------------------------------------------------------------------#
def make_radical_layers(filters_list, in_filters, out_filter):
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

def make_sr1_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 3),
        conv2d(filters_list[0], filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[0], kernel_size=1,
                                        stride=1, padding=0, bias=True),
        nn.AdaptiveAvgPool2d(1)
    ])
    return m


def make_sr2_layers(filters_list, out_filter):
    m = nn.ModuleList([
        nn.Linear(filters_list[0], filters_list[1]),
        nn.Linear(filters_list[1], filters_list[0]),
        nn.Linear(filters_list[0], out_filter)
    ])
    return m


class RIEBody(nn.Module):
    def __init__(self, config):
        super(RIEBody, self).__init__()
        self.config = config
        self.backbone = RABn(None)

        out_filters = self.backbone.layers_out_filters           # out_filters : [32, 64, 128, 256, 512, 1024]

        #------------------------------------------------------------------------#
        #   set two Output Projectors
        #------------------------------------------------------------------------#
        final_out_filter0 = len(config["RIE"]["anchors"][0]) * (5 + config["RIE"]["classes"])   # final_out_filter0 = 192
        final_out_filter1 = int(config["RIE"]["s_classes"])  # final_out_filter1 = 6, the number of the structural relations

        self.OPr = make_radical_layers([512, 1024], out_filters[-1], final_out_filter0)
        self.OPs1 = make_sr1_layers([512, 1024], out_filters[-1], final_out_filter1)
        self.OPs2 = make_sr2_layers([512, 1024], final_out_filter1)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            # print("last_layer", last_layer)
            # print("layer_in", layer_in.shape)

            for i, e in enumerate(last_layer):
                # print("i",i)
                # print("e",e)
                layer_in = e(layer_in)
                # print("layer_in_i", layer_in.shape)
                if i == 2:
                    out_branch = layer_in
            return layer_in, out_branch

        x0 = self.backbone(x) # 13,13,1024

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0, out0_branch = _branch(self.OPr, x0)
        out11, out11_branch = _branch(self.OPs1, x0)
        out11 = out11.reshape(out11.shape[0], -1)
        out1, out1_branch = _branch(self.OPs2, out11)

        return out0, out1



if __name__ == '__main__':
    input=torch.randn(8,3,416,416)
    model = RIEBody(Config)
    out0, out1=model(input)
    print(out0.shape, out1.shape)

    # model = CBR(inplanes=64, planes=[32,64])
    # model = RAB([1,2])
    # model = RAB([1, 2, 8, 8, 4])
    # out=model(input).cuda
    # print(out.shape)
    
    
