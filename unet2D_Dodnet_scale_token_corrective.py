import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools

import sys, os

in_place = True

class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)




class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        # self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1,1),dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet2D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False):
        self.inplanes = 128
        self.weight_std = weight_std
        super(unet2D, self).__init__()

        # self.conv1 = conv3x3x3(3, 32, stride=[1, 1, 1], weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)

        # self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(2, 2))
        self.add_1 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(4, 4))

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            #conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=(0,0),dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=(2, 2))
        self.upsamplex4 = nn.Upsample(scale_factor=(4, 4))

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

        self.x1_resb_add0 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))
        self.x1_resb_add1 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))


        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            # nn.Conv3d(32, 8, kernel_size=1)
            nn.Conv2d(32, 8, kernel_size=(1, 1))
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )

        self.controller_trilinear = nn.Conv2d(256, 162, kernel_size=1, stride=1, padding=0)

        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, 32 + 32 + 64 + 128 + 256 + 256))
        self.sls_emb = nn.Parameter(torch.randn(1, 4, 32 + 32 + 64 + 128 + 256 + 256))
        self.softmax = nn.Softmax(1)
        self.cossim = nn.CosineSimilarity(dim=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                # conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                #           weight_std=self.weight_std),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride, padding=(0, 0), dilation=1, bias=False)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 8))   #### change the channel
        for i in range(N):
            task_encoding[i, task_id[i].long()]=1
        return task_encoding.cuda()

    def encoding_scale(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 4))   #### change the channel
        for i in range(N):
            task_encoding[i, task_id[i].long()]=1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, input, task_id, scale_id, label):

        now_cls_token = self.cls_emb.repeat(input.shape[0], 1, 1)
        now_sls_token = self.sls_emb.repeat(input.shape[0], 1, 1)
        x_start = 0
        x_end = x_start + 32
        cls_token0 = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token0 = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)

        x_start = x_end
        x_end = x_start + 32
        cls_token1 = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token1 = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)

        x_start = x_end
        x_end = x_start + 64
        cls_token2 = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token2 = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)

        x_start = x_end
        x_end = x_start + 128
        cls_token3 = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token3 = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)

        x_start = x_end
        x_end = x_start + 256
        cls_token4 = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token4 = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)


        x_start = x_end
        x_end = x_start + 256
        cls_token_head = now_cls_token[:, int(task_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)
        sls_token_head = now_sls_token[:, int(scale_id[0]), x_start:x_end].unsqueeze(-1).unsqueeze(-1)


        x = self.conv1(input)

        x = self.layer0(x + cls_token0 + sls_token0)
        skip0 = x

        x = self.layer1(x + cls_token1 + sls_token1)
        skip1 = x

        x = self.layer2(x + cls_token2 + sls_token2)
        skip2 = x

        x = self.layer3(x + cls_token3 + sls_token3)
        skip3 = x

        x = self.layer4(x + cls_token4 + sls_token4)

        x = self.fusionConv(x)

        x_feat = self.GAP(x)

        x_cond = x_feat + cls_token_head + sls_token_head

        params = self.controller_trilinear(x_cond)
        params.squeeze_(-1).squeeze_(-1)

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)         # (32, 128, 256)

        x_decoder_feature = x

        head_inputs = self.precls_conv(x)

        feature_maps = head_inputs

        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)

        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, H, W)

        logits_softmax = self.softmax(logits)

        x_decoder_feature_flatten = x_decoder_feature.flatten(start_dim=2)
        logits_softmax_flatten = logits_softmax.flatten(start_dim=2)

        labels_flatten = label.flatten(start_dim=1)

        Top100 = torch.topk((logits_softmax_flatten[:, 1] * labels_flatten).detach().cpu(), k=100, dim=1).indices

        Topfeature = torch.zeros((x_decoder_feature_flatten.shape[0], x_decoder_feature_flatten.shape[1]))

        Bottom50_onlabel = torch.topk(((1 - logits_softmax_flatten[:, 1]) * labels_flatten).detach().cpu(), k=50,
                                      dim=1).indices
        Bottom50_outlabel = torch.topk((logits_softmax_flatten[:, 1] * (1 - labels_flatten)).detach().cpu(), k=50,
                                       dim=1).indices

        Bottom100 = torch.cat([Bottom50_onlabel, Bottom50_outlabel], dim=1)

        Bottomfeature_100 = torch.zeros((x_decoder_feature_flatten.shape[0], x_decoder_feature_flatten.shape[1], 100))
        Intra_sim = torch.zeros((x_decoder_feature_flatten.shape[0], 100))

        for ki in range(x_decoder_feature_flatten.shape[0]):
            Topfeature[ki] = torch.index_select(x_decoder_feature_flatten[ki].detach().cpu(), 1,
                                                torch.LongTensor(Top100[ki]).cpu()).mean(1)
            Bottomfeature_100[ki] = torch.index_select(x_decoder_feature_flatten[ki].detach().cpu(), 1,
                                                       torch.LongTensor(Bottom100[ki]).cpu())

            'average mean'
            # Bottomfeature[ki] = torch.index_select(x_decoder_feature_flatten[ki].detach().cpu(), 1, torch.LongTensor(Bottom100[ki]).cpu()).mean(1)

            'weighted mean'
            Intra_sim[ki] = self.cossim(Topfeature[ki].unsqueeze(0).cuda(),
                                        Bottomfeature_100[ki].permute([1, 0]).cuda())

        shifted_similarity = (Intra_sim + 1) / 2  # Map [-1, 1] to [0, 1]
        inverse_similarity = 1 - shifted_similarity  # Inverse similarity
        weights = inverse_similarity / inverse_similarity.sum(1).unsqueeze(1)  # Normalize weights

        Bottomfeature = torch.bmm(weights.unsqueeze(1), Bottomfeature_100.permute([0, 2, 1])).squeeze(1)

        sim_score = self.cossim(x_decoder_feature, Topfeature.unsqueeze(-1).unsqueeze(-1).cuda())
        sim_score_bottom = self.cossim(x_decoder_feature, Bottomfeature.unsqueeze(-1).unsqueeze(-1).cuda())

        confidence_score = logits_softmax[:, 1]

        return logits, (sim_score - sim_score.min()) / (sim_score.max() - sim_score.min()), (sim_score_bottom - sim_score_bottom.min()) / (sim_score_bottom.max() - sim_score_bottom.min()),  confidence_score, Topfeature, Bottomfeature


def UNet2D(num_classes=1, weight_std=False):
    print("Using DynConv 8,8,2")
    model = unet2D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model
