

# import torch
# import torch.nn as nn
#
# from .darknet import CSPDarknet
# from .network_blocks import BaseConv, CSPLayer, DWConv
#
#
# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """
#
#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv
#
#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat
#
#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )
#
#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )
#
#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )
#
#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.
#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """
#
#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x2, x1, x0] = features
#
#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16
#
#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8
#
#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16
#
#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
#
#         outputs = (pan_out2, pan_out1, pan_out0)
#         return outputs





import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, GhostConv
import torch.utils.data
from .attention import CBAM, SE ,ECA, CAM # 1、导入注意力机制模块

from .mobilenetv3 import mobilenet_v3
from .ghostnet import ghostnet
from .ghostnetv2 import ghostnetv2
from .efficientnet import EfficientNet as EffNet
from .fasternet import fasternet_s
from .fasternet2 import FasterNet
import numpy as np

def updateBN(model):
    for m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.001*torch.sign(m.weight.data))

class FasterNet1(nn.Module):
    def __init__(self):
        super(FasterNet1, self).__init__()
        self.model = FasterNet()

    def forward(self, x):
        model = FasterNet1()

        # x = self.model.patch_embed(x)
        # x1 = self.model.stages[0](x)
        # x2 = self.model.stages[1](x1)
        # out3 = self.model.stages[2](x2)
        # x3 = self.model.stages[3](out3)
        # out4 = self.model.stages[4](x3)
        # x5 = self.model.stages[5](out4)
        # out5 = self.model.stages[6](x5)



        # x = self.model.patch_embed(x)
        # x1 = self.model.stages[0](x)
        # out3 = self.model.stages[1](x1)
        # x3 = self.model.stages[2](out3)
        # out4 = self.model.stages[3](x3)
        # x5 = self.model.stages[4](out4)
        # out5 = self.model.stages[5](x5)

        updateBN(model)
        return out3, out4, out5

    def forward(self, x):
        model = FasterNet1()
        x1 = self.model.stage1(self.model.embedding(x))
        out3 = self.model.stage2(self.model.merging1(x1))
        out4 = self.model.stage3(self.model.merging2(out3))
        out5 = self.model.stage4(self.model.merging3(out4))
        updateBN(model)
        return out3, out4, out5

class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        model = MobileNetV3()
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        updateBN(model)
        return out3, out4, out5

class GhostNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(GhostNetV2, self).__init__()
        model = ghostnetv2()
        if pretrained:
            state_dict = torch.load("model_data/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [2,4,6,8]:
                feature_maps.append(x)
        return feature_maps[1:]

class GhostNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GhostNet, self).__init__()
        model = ghostnet()
        if pretrained:
            state_dict = torch.load("model_data/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [2,4,6,8]:
                feature_maps.append(x)
        return feature_maps[1:]

class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        out_feats = [feature_maps[2],feature_maps[3],feature_maps[4]]
        return out_feats

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        pretrained=False,
        phi=0,
        load_weights=False
    ):
        super().__init__()
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        # self.backbone = MobileNetV3(pretrained=pretrained)
        self.backbone = FasterNet1()
        # self.backbone = EfficientNet(phi, load_weights=load_weights)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = GhostConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = GhostConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = GhostConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = GhostConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )


        # 2、在dark3、dark4、dark5分支后加入CBAM 模块（该分支是主干网络传入FPN的过程中）
        # in_channels = [256, 512, 1024],forward从dark5开始进行，所以cbam_1为dark5
        self.cbam_1 = CBAM(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道
        self.cbam_2 = CBAM(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
        self.cbam_3 = CBAM(int(in_channels[0] * width))  # 对应dark3输出的256维度通道
        # ################


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        # #  backbone
        # out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        out_features = self.backbone(input)
        [x2, x1, x0] = out_features

        # 3、直接对输入的特征图使用注意力机制
        x0 = self.cbam_1(x0)
        x1 = self.cbam_2(x1)
        x2 = self.cbam_3(x2)
        # #################################


        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
