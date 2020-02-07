import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from libs.resnet import resnet18, resnet50

class encoder3(nn.Module):
    def __init__(self, reduce = False):
        super(encoder3,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56
        self.reduce = reduce
        if reduce:
            self.downsample = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                                nn.Conv2d(256,256,1,1,0),
                                nn.ReLU(inplace=True))

    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out,pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out,pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        if self.reduce:
            out = self.downsample(out)
        return out

class decoder3(nn.Module):
    def __init__(self, cls=False, cls_num=32, reduce = False):
        """
        INPUTS:
         - cls: if using classification.
         - cls_num: cluster number
        """
        super(decoder3,self).__init__()
        if reduce:
            self.upsample = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(256,256,1,1,0),
                    nn.ReLU(inplace=True))
        self.reduce = reduce

        self.cls = cls
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,128,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        if not cls:
            self.conv11 = nn.Conv2d(64,3,3,1,0)
        else:
            self.conv11 = nn.Sequential(nn.Conv2d(64,cls_num,3,1,0),
                                        nn.LogSoftmax())

    def forward(self,x):
        output = {}
        if self.reduce:
            x = self.upsample(x)
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        if not self.cls:
            out = F.tanh(out)
        return out


def encoder_res18(pretrained = True, uselayer=3):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet18(uselayer=uselayer)
    if pretrained:
        print("Using pretrianed ResNet18 as guide.")
        model.load_state_dict(torch.load("ae_models/resnet18-5c106cde.pth"), strict = False)
    return model

def encoder_res50(pretrained = True, uselayer=3):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet50(uselayer=uselayer)
    if pretrained:
        print("Using pretrianed ResNet50 as guide.")
        model.load_state_dict(torch.load("ae_models/resnet50-19c8e357.pth"), strict = False)
    return model