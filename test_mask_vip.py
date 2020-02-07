import os
import cv2
import glob
import copy
import queue
import torch
import shutil
import argparse
import scipy.misc
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils

from libs.test_utils import *
from libs.model import transform
from libs.vis_utils import norm_mask
import libs.transforms_pair as transforms
from libs.model import Model_switchGTfixdot_swCC_Res as Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-o","--out_dir",type=str,default="output/",
                        help='output path')
    parser.add_argument("--device", type=int, default=5,
                        help="0~4 for single GPU, 5 for dataparallel.")
    parser.add_argument("-c","--checkpoint_dir",type=str,
                        default="weights/track_Res18_256/checkpoint_latest.pth.tar",
                        help='checkpoints path')
    parser.add_argument('--scale_size', type=int, nargs='+',
                        help='scale size, either a single number for short edge, or a pair for height and width')
    parser.add_argument("--pre_num",type=int,default=7,
                        help='preceding frame numbers')
    parser.add_argument("--temp",type=float,default=1,
                        help='softmax temperature')
    parser.add_argument("--topk",type=int,default=5,
                        help='accumulate label from top k neighbors')
    parser.add_argument("-d", "--root", type=str, default="",
                        help='davis dataset path')
    parser.add_argument("--val_txt", type=str, default="",
                        help='davis evaluation video list')

    print("Begin parser arguments.")
    args = parser.parse_args()
    args.is_train = False

    args.multiGPU = args.device == 5
    if not args.multiGPU:
        torch.cuda.set_device(args.device)

    return args

def transform_topk(aff, frame1, k):
    """
    INPUTS:
     - aff: affinity matrix, b * N * N
     - frame1: reference frame
     - k: only aggregate top-k pixels with highest aff(j,i)
    """
    b,c,h,w = frame1.size()
    b, N, _ = aff.size()
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim = 1, k=k)
    # b * N
    tk_val_min,_ = torch.min(tk_val,dim=1)
    tk_val_min = tk_val_min.view(b,1,N)
    aff[tk_val_min > aff] = 0
    frame1 = frame1.view(b,c,-1)
    frame2 = torch.bmm(frame1, aff)
    return frame2.view(b,c,h,w)

def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list

def create_transforms():
    normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
    t = []
    t.extend([transforms.ToTensor(),
    		  normalize])
    return transforms.Compose(t)

def read_frame(frame_dir, transforms):
    frame = cv2.imread(frame_dir)
    ori_h,ori_w,_ = frame.shape
    if(len(args.scale_size) == 1):
        if(ori_h > ori_w):
        	tw = args.scale_size[0]
        	th = (tw * ori_h) / ori_w
        	th = int((th // 64) * 64)
        else:
        	th = args.scale_size[0]
        	tw = (th * ori_w) / ori_h
        	tw = int((tw // 64) * 64)
    else:
        tw = args.scale_size[1]
        th = args.scale_size[0]
    frame = cv2.resize(frame, (tw,th))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    pair = [frame, frame]
    transformed = list(transforms(*pair))
    return transformed[0].cuda().unsqueeze(0), ori_h, ori_w

def forward(frame1, frame2, model, seg):
    n, c, h, w = frame1.size()
    frame1_gray = frame1[:,0].view(n,1,h,w)
    frame2_gray = frame2[:,0].view(n,1,h,w)
    frame1_gray = frame1_gray.repeat(1,3,1,1)
    frame2_gray = frame2_gray.repeat(1,3,1,1)

    output = model(frame1_gray, frame2_gray, frame1, frame2)
    aff = output[2]

    frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)

    return frame2_seg.cpu()

def test(model, frame_list, first_seg):
    transforms = create_transforms()

    # The queue stores `pre_num` preceding frames
    que = queue.Queue(args.pre_num)

    # frame 1
    frame1, ori_h, ori_w = read_frame(frame_list[0], transforms)
    n, c, h, w = frame1.size()

    for cnt in tqdm(range(1,len(frame_list))):
        frame_tar, ori_h, ori_w = read_frame(frame_list[cnt], transforms)

        # from first to t
        with torch.no_grad():
            frame_tar_acc = forward(frame1, frame_tar, model, first_seg)

            # previous 7 frames
            tmp_queue = list(que.queue)
            for pair in tmp_queue:
                framei = pair[0]
                segi = pair[1]
                frame_tar_est_i = forward(framei, frame_tar, model, segi)
                frame_tar_acc += frame_tar_est_i
            frame_tar_avg = frame_tar_acc / (1 + len(tmp_queue))

        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg",".png")
        out_path = os.path.join(video_folder,frame_nm)

        # pop out oldest frame if neccessary
        # push current result into queue
        if(que.qsize() == args.pre_num):
            que.get()
        seg = copy.deepcopy(frame_tar_avg)
        frame, ori_h, ori_w = read_frame(frame_list[cnt], transforms)
        que.put([frame,seg])

        # upsampling & argmax
        frame_tar_avg = torch.nn.functional.interpolate(frame_tar_avg,scale_factor=8,mode='bilinear')
        frame_tar_avg = norm_mask(frame_tar_avg.squeeze())
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        frame_tar_seg = frame_tar_seg.squeeze().numpy()
        frame_tar_seg = np.array(frame_tar_seg, dtype=np.uint8)
        frame_tar_seg = scipy.misc.imresize(frame_tar_seg, (ori_h,ori_w),"nearest",mode="F")
        imwrite_indexed(out_path, np.uint8(frame_tar_seg))

def to_one_hot(y_tensor, n_dims=None):
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2,0,1).unsqueeze(0)

def read_seg(seg_dir):
    seg = Image.open(seg_dir)
    h,w = seg.size
    if(len(args.scale_size) == 1):
        if(h > w):
        	tw = args.scale_size[0]
        	th = (tw * h) / w
        	th = int((th // 64) * 64)
        else:
        	th = args.scale_size[0]
        	tw = (th * w) / h
        	tw = int((tw // 64) * 64)
    else:
        tw = args.scale_size[1]
        th = args.scale_size[0]
    seg = np.asarray(seg).reshape((w,h,1))
    seg = np.squeeze(seg)
    seg = scipy.misc.imresize(seg, (tw//8,th//8),"nearest",mode="F")

    t = []
    t.extend([transforms.ToTensor()])
    trans = transforms.Compose(t)
    pair = [seg, seg]
    transformed = list(trans(*pair))
    seg = transformed[0]
    return to_one_hot(seg)

if(__name__ == '__main__'):
    args = parse_args()
    with open(args.val_txt) as f:
        lines = f.readlines()
    f.close()

    model = Model(pretrainRes=False, temp = args.temp, uselayer=4)
    if(args.multiGPU):
        model = nn.DataParallel(model)
    print("=> loading checkpoint '{}'".format(args.checkpoint_dir))
    checkpoint = torch.load(args.checkpoint_dir)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{} ({})' (epoch {})"
          .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
    model.cuda()
    model.eval()

    for cnt in range(0,len(lines)):
        line = lines[cnt]
        tmp = line.strip().split('/')[-1]
        video_nm = tmp
        print('[{:n}/{:n}] Begin to segment video {}.'.format(cnt,len(lines),video_nm))

        video_dir = os.path.join(args.root, video_nm)
        frame_list = read_frame_list(video_dir)
        seg_dir = frame_list[0].replace("Images","Annotations/Category_ids")
        seg_dir = seg_dir.replace("jpg","png")
        first_seg = read_seg(seg_dir)

        # include first frame in testing
        video_dir = os.path.join(video_dir)
        video_folder = os.path.join(args.out_dir, video_nm)
        os.makedirs(video_folder, exist_ok = True)

        seg_vis = Image.open(seg_dir)
        seg_vis = np.array(seg_vis)
        out_path = os.path.join(video_folder, 'output_001.png')
        imwrite_indexed(out_path,np.uint8(seg_vis))

        first_seg_nm = seg_dir.split('/')[-1]
        shutil.copy(seg_dir, os.path.join(video_folder,first_seg_nm))
        test(model, frame_list, first_seg)
