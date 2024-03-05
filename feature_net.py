import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()
        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)
    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]

class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.2, part=3, alpha=0.2, nheads=4, arch='resnet50', wpa=False):
        super(embed_net, self).__init__()
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 2048, 2)

    def forward(self, x1, x2, p1, p2, modal=0, seq_len=12):
        b, c, h, w = x1.size()
        t = seq_len

        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        x = self.base_resnet(x)

        x_h = self.avgpool(x).squeeze()
        x_h = x_h.view(x_h.size(0) // t, t, -1).permute(1, 0, 2)
        output, _ = self.lstm(x_h)
        t = output[-1]
        # x_pool=torch.cat(t,x_h)

        feat = self.bottleneck(t)
        # 如果处于训练阶段，返回融合后的特征向量和全连接层输出。
        # 如果处于测试阶段，返回经过L2范数归一化后的特征向量。
        if self.training:
            return t, self.classifier(feat)
        else:
            return self.l2norm(feat)


if __name__ == '__main__':
    # 进行一个网络的测试
    input1 = torch.randn(16, 36, 288, 144)
    input2 = torch.randn(16, 36, 288, 144)
    input3 = torch.randn(16, 12, 19, 6)
    input4 = torch.randn(16, 12, 19, 6)
    net = embed_net(512, 500, drop=0.2, part=3, alpha=0.2, nheads=4, arch='resnet50', wpa=False)

    output_pool, output_cls = net(input1, input2, input3, input4, seq_len=12)
    print(output_pool.shape)
    print(output_cls.shape)
