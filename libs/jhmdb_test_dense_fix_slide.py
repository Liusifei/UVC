from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc



# get the video frames
# two patches in the future frame, one is center, the other is one of the 8 patches around

class JHMDBSet(data.Dataset):
    def __init__(self, params, is_train=True, sigma=0.5, resnet=False):

        self.filelist = params['filelist']
        self.batchSize = params['batchSize']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']
        self.predFrames = params['predFrames'] # 4
        self.sideEdge = params['sideEdge'] # 64

        self.sigma = sigma
        self.resnet = resnet


        # prediction distance, how many frames far away
        self.predDistance = params['predDistance']
        # offset x,y parameters
        self.offset = params['offset']
        # gridSize = 3
        # self.gridSize = params['gridSize']

        self.is_train = is_train

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[1]
            lblfile = rows[0]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def cropimg(self, img, offset_x, offset_y, cropsize):

        img = im_to_numpy(img)
        cropim = np.zeros([cropsize, cropsize, 3])
        cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
        cropim = im_to_torch(cropim)

        return cropim


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        patches = []
        target_imgs = []

        folder_contains = os.listdir(folder_path)
        pngcnt = 0
        for i in range(len(folder_contains)):
            if '.png' in folder_contains[i]:
                pngcnt = pngcnt + 1

        frame_num = pngcnt + self.videoLen

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        ht = 0
        wd = 0

        for i in range(frame_num):
            if i < self.videoLen:
                img_path = folder_path + "/{:05d}.png".format(1)
            else:
                img_path = folder_path + "/{:05d}.png".format(i - self.videoLen + 1)

            if(self.resnet):
                img = load_image(img_path)  # CxHxW
            else:
                img = load_image_lab(img_path)  # CxHxW
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd

            if ht <= wd:
                ratio  = 1.0 #float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = 1.0 #float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            if i == 0:
                imgs = torch.Tensor(frame_num, 3, newh, neww)

            img = color_normalize(img, mean, std)
            imgs[i] = img
            # lblimg  = scipy.misc.imread(lbl_path)
            # lblimg  = scipy.misc.imresize( lblimg, (newh, neww), 'nearest' )
            # lbls.append(lblimg.copy())


        lbls_mat = sio.loadmat(label_path)

        lbls_coord = lbls_mat['pos_img']
        lbls_coord = lbls_coord - 1

        #import pdb; pdb.set_trace()

        lbls_coord[0, :, :] = lbls_coord[0, :, :] * float(neww) / float(wd) / 8.0
        lbls_coord[1, :, :] = lbls_coord[1, :, :] * float(newh) / float(ht) / 8.0
        lblsize =  int(self.cropSize / 8.0)

        lbls = np.zeros((lbls_coord.shape[2], lblsize, lblsize, lbls_coord.shape[1]))

        if lbls_coord.shape[2] != pngcnt:
            print(folder_path)

        for i in range(lbls_coord.shape[2]):
            lbls_coord_now = lbls_coord[:, :, i]

            for j in range(lbls_coord.shape[1]):
                if self.sigma > 0:
                    draw_labelmap_np(lbls[i, :, :, j], lbls_coord_now[:, j], self.sigma)
                else:
                    tx = int(lbls_coord_now[0, j])
                    ty = int(lbls_coord_now[1, j])
                    if tx < lblsize and ty < lblsize and tx >=0 and ty >=0:
                        lbls[i, ty, tx, j] = 1.0


        lbls_tensor = torch.zeros(frame_num, lblsize, lblsize, lbls_coord.shape[1])

        for i in range(frame_num):
            if i < self.videoLen:
                nowlbl = lbls[0]
            else:
                if(i - self.videoLen < len(lbls)):
                    nowlbl = lbls[i - self.videoLen]
            lbls_tensor[i] = torch.from_numpy(nowlbl)

        # lbls_tensor = torch.from_numpy(lbls)



        gridx = 0
        gridy = 0

        for i in range(frame_num):

            img = imgs[i]
            newh, neww = img.size(1), img.size(2)

            sideEdge = self.sideEdge

            gridy = int(newh / sideEdge)
            gridx = int(neww / sideEdge)

            # img = im_to_numpy(img)
            # target_imgs.append(img)

            for yid in range(gridy):
                for xid in range(gridx):

                    patch_img = img[:, yid * sideEdge: yid * sideEdge + sideEdge, xid * sideEdge: xid * sideEdge + sideEdge].clone()
                    # patch_img = im_to_torch(patch_img)
                    # patch_img = resize(patch_img, self.cropSize2, self.cropSize2)
                    # patch_img = color_normalize(patch_img, mean, std)

                    patches.append(patch_img)


        countPatches = frame_num * gridy * gridx
        patchTensor = torch.Tensor(countPatches, 3, self.cropSize2, self.cropSize2)

        for i in range(countPatches):
            patchTensor[i, :, :, :] = patches[i]

        # for i in range(len(imgs)):
        #     imgs[i] = color_normalize(imgs[i], mean, std)


        patchTensor = patchTensor.view(frame_num, gridy * gridx, 3, self.cropSize2, self.cropSize2)

        # Meta info
        meta = {'folder_path': folder_path, 'gridx': gridx, 'gridy': gridy, 'ht': int(ht), 'wd': int(wd)}


        # lbls_tensor = torch.Tensor(len(lbls), newh, neww, 3)
        # for i in range(len(lbls)):
        #     lbls_tensor[i] = torch.from_numpy(lbls[i])


        return imgs, patchTensor, lbls_tensor, meta

    def __len__(self):
        return len(self.jpgfiles)
