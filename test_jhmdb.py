'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import libs.model as video3d

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

import libs.jhmdb_test_dense_fix_slide as jhmdb

# tps model
#from model.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric, FeatureCorrelation, featureL2Norm
#from model.loss import TransformedGridLoss, WeakInlierCount, TwoStageWeakInlierCount

#from geotnf.transformation import SynthPairTnf,SynthTwoPairTnf,SynthTwoStageTwoPairTnf
#from geotnf.transformation import GeometricTnf

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict
from utils.torch_util import expand_dim

import matplotlib.pyplot as plt
import matplotlib
import math

params = {}
params['filelist'] = 'testlist_split1.txt'
# params['batchSize'] = 24
params['imgSize'] = 240
params['cropSize'] = 240
params['cropSize2'] = 80
params['videoLen'] = 8
params['offset'] = 0
params['sideEdge'] = 80
params['predFrames'] = 1



def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
"""
parser.add_argument('-c', '--checkpoint', default='/scratch/xiaolonw/pytorch_checkpoints/unsup3dnl_single_contrast', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
"""
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--predDistance', default=0, type=int,
                    help='predict how many frames away')
parser.add_argument('--seperate2d', type=int, default=0, help='manual seed')
parser.add_argument('--batchSize', default=1, type=int,
                    help='batchSize')
parser.add_argument('--T', default=1.0, type=float,
                    help='temperature')
parser.add_argument('--gridSize', default=9, type=int,
                    help='temperature')
parser.add_argument('--classNum', default=49, type=int,
                    help='temperature')
parser.add_argument('--lamda', default=0.1, type=float,
                    help='temperature')
parser.add_argument('--use_softmax', type=str_to_bool, nargs='?', const=True, default=True,
                    help='pretrained_imagenet')
parser.add_argument('--use_l2norm', type=str_to_bool, nargs='?', const=True, default=False,
                    help='pretrained_imagenet')
parser.add_argument('--pretrained_imagenet', type=str_to_bool, nargs='?', const=True, default=True,
                    help='pretrained_imagenet')
parser.add_argument('--topk_vis', default=1, type=int,
                    help='topk_vis')

parser.add_argument('--videoLen', default=8, type=int,
                    help='predict how many frames away')
parser.add_argument('--frame_gap', default=2, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=240, type=int,
                    help='predict how many frames away')
parser.add_argument('--cropSize2', default=80, type=int,
                    help='predict how many frames away')
parser.add_argument('--temporal_out', default=4, type=int,
                    help='predict how many frames away')
parser.add_argument('--sigma', default=0.5, type=float,
                    help='temperature')

parser.add_argument('--resnet', action='store_true',
                    help='test on resnet')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

params['predDistance'] = state['predDistance']
print(params['predDistance'])

params['batchSize'] = state['batchSize']
print('batchSize: ' + str(params['batchSize']) )

print('temperature: ' + str(state['T']))

params['gridSize'] = state['gridSize']
print('gridSize: ' + str(params['gridSize']) )

params['classNum'] = state['classNum']
print('classNum: ' + str(params['classNum']) )

params['videoLen'] = state['videoLen']
print('videoLen: ' + str(params['videoLen']) )

params['cropSize'] = state['cropSize']
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = state['cropSize']


params['cropSize2'] = state['cropSize2']
print('cropSize2: ' + str(params['cropSize2']) )
params['sideEdge'] = state['cropSize2']


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

print(args.gpu_id)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


palette_list = 'palette.txt'
f = open(palette_list, 'r')
palette = np.zeros((256, 3))
cnt = 0
for line in f:
    rows = line.split()
    palette[cnt][0] = int(rows[0])
    palette[cnt][1] = int(rows[1])
    palette[cnt][2] = int(rows[2])
    cnt = cnt + 1

f.close()
palette = palette.astype(np.uint8)



def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    val_loader = torch.utils.data.DataLoader(
        jhmdb.JHMDBSet(params, is_train=False, sigma=args.sigma, resnet=args.resnet),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = video3d.CycleTime(class_num=params['classNum'], trans_param_num=3,
                              pretrained=args.pretrained_imagenet,
                              temporal_out=args.temporal_out,
                              use_resnet=args.resnet)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # from IPython import embed; embed()
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    title = 'videonet'
    """
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # contrastnet.load_state_dict(checkpoint['contrast_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])

        del checkpoint

    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])
    """

    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(val_loader, model, criterion, 1, use_cuda)
        print(' Test Loss:  %.8f' % (test_loss))
        return



def vis(oriImg, points):

    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas


    # plt.imshow(canvas[:, :, [2, 1, 0]])
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)

    # from time import gmtime, strftime
    # import os
    # directory = 'data/mpii/result/test_images'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # fn = os.path.join(directory, strftime("%Y-%m-%d-%H_%M_%S", gmtime()) + '.jpg')

    # plt.savefig(fname)



def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    save_objs = args.evaluate
    save_path = 'jhmdb/'
    save_file = 'jhmdb/list.txt'
    os.makedirs(save_path, exist_ok = True)

    fileout = open(save_file, 'w')

    end = time.time()

    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (imgs_total, patch2_total, lbls, meta) in enumerate(val_loader):
        print('===> Processing {:3n} video.'.format(batch_idx))

        finput_num_ori = params['videoLen']
        finput_num     = finput_num_ori

        # measure data loading time
        data_time.update(time.time() - end)
        imgs_total = imgs_total.cuda()

        bs = imgs_total.size(0)
        total_frame_num = imgs_total.size(1)
        channel_num = imgs_total.size(2)
        height_len  = imgs_total.size(3)
        width_len   = imgs_total.size(4)

        assert(bs == 1)

        folder_paths = meta['folder_path']
        print('total_frame_num: ' + str(total_frame_num))

        height_dim = int(params['cropSize'] / 8)
        width_dim  = int(params['cropSize'] / 8)

        # processing labels
        lbls = lbls[0].data.cpu().numpy()

        lbl_set = palette[0: 16, :]
        lbls_resize2 = np.zeros((lbls.shape[0], height_dim, width_dim, len(lbl_set)))

        for i in range(lbls.shape[0]):
            nowlbl = lbls[i].copy()
            lbls_resize2[i, :, :, 1 :] = nowlbl
            for h in range(height_dim):
                for w in range(width_dim):
                    lblsum = lbls_resize2[i, h, w, :].sum()
                    if lblsum == 0:
                        lbls_resize2[i, h, w, 0] = 1.0


        imgs_set = imgs_total.data
        imgs_set = imgs_set.cpu().numpy()
        imgs_set = imgs_set[0]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        imgs_toprint = []

        # ref image
        for t in range(imgs_set.shape[0]):
            img_now = imgs_set[t]

            for c in range(3):
                img_now[c] = img_now[c] * std[c]
                img_now[c] = img_now[c] + mean[c]

            img_now = img_now * 255
            img_now = np.transpose(img_now, (1, 2, 0))
            img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )

            imgs_toprint.append(img_now)

            imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
            scipy.misc.imsave(imname, img_now)


        now_batch_size = 1

        imgs_stack = []
        patch2_stack = []

        im_num = total_frame_num - finput_num_ori
        all_coord = np.zeros((2, 15, total_frame_num - finput_num_ori))

        trans_out_2_set = []
        corrfeat2_set = []

        imgs_tensor = torch.Tensor(now_batch_size, finput_num, 3, params['cropSize'], params['cropSize'])
        target_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize'])
        imgs_tensor = torch.autograd.Variable(imgs_tensor.cuda())
        target_tensor = torch.autograd.Variable(target_tensor.cuda())


        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                imgs = imgs_total[:, iter + i + 1: iter + i + finput_num_ori, :, :, :]
                imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
                imgs = torch.cat((imgs2, imgs), dim=1)

                imgs_tensor[i] = imgs
                target_tensor[i, 0] = imgs_total[0, iter + i + finput_num_ori]

            with torch.no_grad():
                corrfeat2_now = model(imgs_tensor, target_tensor)
                corrfeat2_now = corrfeat2_now.view(now_batch_size, finput_num_ori, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))

            #for i in range(now_batch_size2):
            #    corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())


        #for iter in range(total_frame_num - finput_num_ori):

            #if iter % 10 == 0:
                #print(iter)

            imgs = imgs_total[:, iter + 1: iter + finput_num_ori, :, :, :]
            imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
            imgs = torch.cat((imgs2, imgs), dim=1)

            #corrfeat2   = corrfeat2_set[iter]
            #corrfeat2   = torch.from_numpy(corrfeat2)
            corrfeat2 = corrfeat2_now[i].cpu()

            out_frame_num = int(finput_num)
            height_dim = corrfeat2.size(2)
            width_dim = corrfeat2.size(3)

            corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
            corrfeat2 = corrfeat2.data.cpu().numpy()

            topk_vis = args.topk_vis
            vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
            vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

            predlbls = np.zeros((height_dim, width_dim, len(lbl_set)))

            for t in range(corrfeat2.shape[0]):
                for h in range(height_dim):
                    for w in range(width_dim):
                        attenmap = corrfeat2[t, :, :, h, w]
                        atten1d  = attenmap.reshape(height_dim * width_dim)
                        ids = np.argsort(atten1d)
                        ids = ids[- topk_vis:]

                        for k in range(len(ids)):
                            nowid = ids[k]

                            hid   = int(nowid / width_dim)
                            wid   = int(nowid % width_dim)

                            vis_ids_h[t, h, w, k] = wid # hid
                            vis_ids_w[t, h, w, k] = hid # wid

                            if t == 0:
                                lbl = lbls_resize2[0, hid, wid, :]
                            else:
                                lbl = lbls_resize2[t + iter, hid, wid, :]

                            predlbls[h, w, :] += lbl * corrfeat2[t, hid, wid, h, w]

            img_now = imgs_toprint[iter + finput_num_ori]

            predlbls = predlbls / finput_num

            # from IPython import embed; embed()

            for t in range(len(lbl_set)):
                nowt = t
                if np.sum(predlbls[:, :, nowt])  == 0:
                    continue

                predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
                predlbls[:, :, nowt] = predlbls[:, :, nowt] / predlbls[:, :, nowt].max()


            # generate the coordinate:
            current_coord = np.zeros((2, len(lbl_set) - 1))
            top_ids = 5
            for t in range(len(lbl_set) - 1):
                pred = predlbls[:, :, t + 1]
                pred = pred.reshape(-1)
                if pred.sum() == 0:
                    current_coord[:, t] = -1
                    continue
                ids = np.argsort(pred)[-top_ids:]
                vals = np.sort(pred)[-top_ids:]

                vals = vals / vals.sum()

                for i in range(len(ids)):
                    nowid = ids[i]
                    nowx  = nowid % width_dim
                    nowy  = int(nowid / width_dim)
                    current_coord[0, t] += nowx * vals[i]
                    current_coord[1, t] += nowy * vals[i]

            all_coord[:, :, iter] = current_coord

            current_coord_scaleup = current_coord.copy()
            current_coord_scaleup = current_coord_scaleup * 16



            lbls_resize2[iter + finput_num_ori] = predlbls
            predlbls_cp = predlbls.copy()
            predlbls_val = np.zeros((height_dim, width_dim, 3))

            for j in range(height_dim):
                for k in range(width_dim):
                    nowid = np.argmax(predlbls_cp[j, k, :])
                    predlbls_val[j, k, :] = lbl_set[nowid]

            predlbls_val = cv2.resize(predlbls_val, (params['imgSize'], params['imgSize']))
            predlbls_val_sharp = np.zeros((height_dim, width_dim, 3))

            ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)


            for t in range(len(lbl_set) - 1):
                x = int(current_coord[0, t])
                y = int(current_coord[1, t])

                if x >=0 and y >= 0:
                    predlbls_val_sharp[y, x, :] = lbl_set[t + 1]


            predlbls_val = predlbls_val.astype(np.uint8)
            predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)
            img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

            predlbls_val_sharp = predlbls_val_sharp.astype(np.uint8)
            predlbls_val_sharp2 = cv2.resize(predlbls_val_sharp, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)
            img_with_heatmap2 =  np.float32(img_now) * 0.5 + np.float32(predlbls_val_sharp2) * 0.5


            imname  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_label.jpg'
            imname2  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_mask.png'
            imname3  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_label2.jpg'
            imname4  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_pose.jpg'

            scipy.misc.imsave(imname, np.uint8(img_with_heatmap))
            scipy.misc.imsave(imname3, np.uint8(img_with_heatmap2))
            scipy.misc.imsave(imname2, np.uint8(predlbls_val))

            img_now_vis = vis(img_now, current_coord_scaleup)
            scipy.misc.imsave(imname4, np.uint8(img_now_vis))


        coordname  = save_path + str(batch_idx) + '.dat'
        all_coord.dump(coordname)



    fileout.close()


    return losses.avg

if __name__ == '__main__':
    main()
