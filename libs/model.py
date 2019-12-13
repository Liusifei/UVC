# from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
# from correlation_package.modules.corr import Correlation
import math
import copy
import numpy as np
from . import resnet_res4s1
from . import inflated_resnet
from .autoencoder import encoder_res18
# import i3res_res3
import torchvision

import torch.nn.functional as F
#from geotnf.transformation import GeometricTnf,GeometricTnfAffine
#from geotnf.loss import TransformedGridLoss, WeakInlierCountPool
from utils.torch_util import expand_dim

import random
import utils.imutils2


import time
import sys

import torchvision.utils as vutils

class CycleTime(nn.Module):

    def __init__(self, class_num=8, dim_in=2048, trans_param_num=6, detach_network=False,
                 pretrained=True, temporal_out=4, T=None, use_resnet=False):
        super(CycleTime, self).__init__()

        dim = 512
        print(pretrained)

        if(use_resnet):
            #resnet = resnet_res4s1.resnet50(pretrained=True)
            resnet = resnet_res4s1.resnet18(pretrained=True)
        else:
            resnet = encoder_res18(uselayer=4)
            resnet.load_state_dict(torch.load('weights/encoder.pth.tar'))
            print('Loading from dropbox')
            #resnet.load_state_dict(torch.load('/home/xtli/Dropbox/Neurips2019/switch_nopre_lc0.pth.tar'))
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))
        self.detach_network = detach_network

        #self.div_num = 512
        if(use_resnet):
            self.div_num = 512
        else:
            self.div_num = 1
        self.T = self.div_num**-.5 if T is None else T

        print('self.T:', self.T)

        # self.encoderVideo = resnet3d.resnet50(pretrained=False)

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

        self.spatial_out1 = 30
        self.temporal_out = temporal_out

        #self.afterconv3_trans = nn.Conv2d(self.spatial_out1 * self.spatial_out1, 128, kernel_size=4, padding=0, bias=False)
        #self.afterconv4_trans = nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False)

        corrdim = 64 * 4 * 4
        corrdim_trans = 64 * 4 * 4

        self.linear2 = nn.Linear(corrdim_trans, trans_param_num)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.avgpool3d = nn.AvgPool3d((4, 2, 2), stride=(1, 2, 2))
        self.maxpool2d = nn.MaxPool2d(2, stride=2)

        self.use_resnet = use_resnet


        # initialization

        #nn.init.kaiming_normal_(self.afterconv1.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.afterconv3_trans.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.afterconv4_trans.weight, mode='fan_out', nonlinearity='relu')

        # assuming no fc pre-training
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight.data.normal_(0, 0.01)
        #        m.bias.data.zero_()


        xs = np.linspace(-1,1,80)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        self.xs = xs


    def compute_corr_softmax(self, patch_feat1, r50_feat2, finput_num, spatial_out1, spatial_out2):
        if(self.use_resnet):
            print('use resent')

        #r50_feat2 = r50_feat2.transpose(3, 4) # for the inlier counter
        #r50_feat2 = r50_feat2.contiguous()
        #print(r50_feat2.size(), patch_feat1.size())
        r50_feat2_vec = r50_feat2.view(r50_feat2.size(0), r50_feat2.size(1), -1)
        r50_feat2_vec = r50_feat2_vec.transpose(1, 2)

        patch_feat1_vec = patch_feat1.view(patch_feat1.size(0), patch_feat1.size(1), -1)
        #corrfeat = torch.matmul(r50_feat2_vec, patch_feat1_vec)
        corrfeat = torch.bmm(r50_feat2_vec, patch_feat1_vec)

        # if self.use_l2norm is False:
        corrfeat = torch.div(corrfeat, self.div_num**-.5)

        corrfeat  = corrfeat.view(corrfeat.size(0), finput_num, spatial_out1 * spatial_out2, spatial_out1, spatial_out2)
        corrfeat  = F.softmax(corrfeat, dim=2)
        corrfeat  = corrfeat.view(corrfeat.size(0), finput_num * spatial_out1 * spatial_out2, spatial_out1, spatial_out2)

        return corrfeat


    def forward(self, ximg1, img2, retfeats=False):

        bs = ximg1.size(0)
        finput_num = ximg1.size(1)

        # ximg1: 4 x 8 x 3 x 320 x 320
        # 32 x 3 x 320 x 320
        ximg1_images = ximg1.view(ximg1.size(0) * ximg1.size(1), ximg1.size(2), ximg1.size(3), ximg1.size(4)).clone()

        videoclip1  = ximg1

        # video feature clip1
        videoclip1 = videoclip1.transpose(1, 2) # 4 x 3 x 8 x 320 x 320
        r50_feat1 = self.encoderVideo(videoclip1, self.use_resnet) # 4 x 1024 x 8 x 40 x 40

        if self.detach_network is True:
            r50_feat1 = r50_feat1.detach()

        #r50_feat1 = self.afterconv1(r50_feat1)
        #r50_feat1_relu = self.relu(r50_feat1)
        # if self.use_softmax is False or self.use_l2norm is True:
        if(self.use_resnet):
            r50_feat1_norm = F.normalize(r50_feat1, p=2, dim=1)
        else:
            r50_feat1_norm = r50_feat1


        # target image feature
        img2 = img2.transpose(1, 2)
        img_feat2_pre = self.encoderVideo(img2, self.use_resnet)

        #img_feat2 = selfself.afterconv1(img_feat2_pre)
        #img_feat2 = self.relu(img_feat2_pre)
        #img_feat2 = img_feat2.contiguous()
        #img_feat2 = img_feat2_pre.view(img_feat2_pre.size(0), img_feat2_pre.size(1), img_feat2.size(3), img_feat2.size(4))
        if(self.use_resnet):
            img_feat2_norm = F.normalize(img_feat2_pre, p=2, dim=1)
        else:
            img_feat2_norm = img_feat2_pre
        img_feat2 = img_feat2_pre

        spatial_out1 = img_feat2.size(3)
        spatial_out2 = spatial_out1
        #spatial_out2 = img_feat2.size(4)

        corrfeat_trans_matrix_target  = self.compute_corr_softmax(img_feat2_norm, r50_feat1_norm, finput_num, spatial_out1, spatial_out2)
        corrfeat_trans_matrix_target = corrfeat_trans_matrix_target.contiguous()
        corrfeat_trans_matrix_target2 = corrfeat_trans_matrix_target.view(corrfeat_trans_matrix_target.size(0) * finput_num, spatial_out1 * spatial_out2, spatial_out1, spatial_out2)

        return corrfeat_trans_matrix_target2
