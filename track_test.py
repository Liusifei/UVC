"""
Test NLM on segmentation mask recursively
Use first frame with gt mask and preceding 7 frames
"""
import os
import cv2
import glob
import copy
import math
import queue
import torch
import argparse
import scipy.misc
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils
from libs.model import transform
from libs.vis_utils import norm_mask
import libs.transforms_pair as transforms
from libs.model import Model_switchGTfixdot_swCC_Res as Model
from libs.track_utils import seg2bbox, draw_bbox, match_ref_tar, bbox_in_tar_scale, squeeze_all, seg2bbox_v2, vis_bbox

import sys
sys.path.append('libs/')
import davis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("-o","--out_dir",type=str,default="track_v2/",
                        help='output path')
    parser.add_argument("--device", type=int, default=5,
                        help="0~4 for single GPU, 5 for dataparallel.")
    parser.add_argument("-c","--checkpoint_dir",type=str,default="weights/model_best.pth.tar",
                        help='checkpoints path')
    parser.add_argument('--scale_size', type=int, nargs='+', default=[1024],
                        help='scale size, either a single number for short edge, or a pair for height and width')
    parser.add_argument("--pre_num",type=int,default=7,
                        help='preceding frame numbers')
    parser.add_argument("--no_rec",action="store_true",
                        help='not using averaged result from preceding frames')
    parser.add_argument("--temp",type=float,default=1,
                        help='softmax temperature')
    parser.add_argument("--topk",type=int,default=5,
                        help='accumulate label from top k neighbors')
    parser.add_argument("--davis_dir", type=str, default="/DATA/DAVIS2017/JPEGImages/480p/",
                        help='davis dataset path')
    parser.add_argument("--val_txt", type=str, default="/DATA/DAVIS2017/ImageSets/2017/val.txt",
                        help='davis evaluation video list')
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet50", "resnet18"])

    print("Begin parser arguments.")
    args = parser.parse_args()
    args.is_train = False

    args.multiGPU = args.device == 5
    if not args.multiGPU:
        torch.cuda.set_device(args.device)

    return args

def transform_topk(aff, frame1, k, h2=None, w2=None):
    """
    INPUTS:
     - aff: affinity matrix, b * N * N
     - frame1: reference frame
     - k: only aggregate top-k pixels with highest aff(j,i)
    """
    b,c,h,w = frame1.size()
    b, N1, N2 = aff.size()
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim = 1, k=k)
    # b * N
    tk_val_min,_ = torch.min(tk_val,dim=1)
    tk_val_min = tk_val_min.view(b,1,N2)
    aff[tk_val_min > aff] = 0
    frame1 = frame1.contiguous().view(b,c,-1)
    frame2 = torch.bmm(frame1, aff)
    if(h2 is None):
        return frame2.view(b,c,h,w)
    else:
        return frame2.view(b,c,h2,w2)

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

def imwrite_indexed(filename,array):
    """ Save indexed png."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

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

def adjust_bbox(bbox_now, bbox_pre, a, h, w):
    for cnt in bbox_pre.keys():
        if(cnt == 0):
            continue
        if(cnt in bbox_now and bbox_pre[cnt] is not None and bbox_now[cnt] is not None):
            bbox_now_h = (bbox_now[cnt].top  + bbox_now[cnt].bottom) / 2.0
            bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0

            bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
            bbox_now_width_  = bbox_now[cnt].right  - bbox_now[cnt].left

            bbox_pre_height = bbox_pre[cnt].bottom - bbox_pre[cnt].top
            bbox_pre_width  = bbox_pre[cnt].right  - bbox_pre[cnt].left

            bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
            bbox_now_width  = a * bbox_now_width_  + (1 - a) * bbox_pre_width

            bbox_now[cnt].left   = math.floor(bbox_now_w - bbox_now_width / 2.0)
            bbox_now[cnt].right  = math.ceil(bbox_now_w + bbox_now_width / 2.0)
            bbox_now[cnt].top    = math.floor(bbox_now_h - bbox_now_height / 2.0)
            bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)

            bbox_now[cnt].left = max(0, bbox_now[cnt].left)
            bbox_now[cnt].right = min(w, bbox_now[cnt].right)
            bbox_now[cnt].top = max(0, bbox_now[cnt].top)
            bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)

    return bbox_now

def bbox_next_frame(img_ref, seg_ref, img_tar, bbox_ref):
    """
    Match bbox from the reference frame to the target frame
    """
    F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    seg_ref = seg_ref.squeeze(0)
    F_ref, F_tar = squeeze_all(F_ref, F_tar)
    c, h, w = F_ref.size()

    # get coordinates of each point in the target frame
    coords_ref_tar = match_ref_tar(F_ref, F_tar, seg_ref, args.temp)
    bbox_tar = bbox_in_tar_scale(coords_ref_tar, bbox_ref, h, w)
    bbox_tar = adjust_bbox(bbox_tar, bbox_ref, 0.1, h, w)
    return bbox_tar, coords_ref_tar

def forward(frame1, frame2, model, seg, return_feature=False):
    n, c, h, w = frame1.size()
    frame1_gray = frame1[:,0].view(n,1,h,w)
    frame2_gray = frame2[:,0].view(n,1,h,w)
    frame1_gray = frame1_gray.repeat(1,3,1,1)
    frame2_gray = frame2_gray.repeat(1,3,1,1)

    output = model(frame1_gray, frame2_gray, frame1, frame2)
    if(return_feature):
        return output[-2], output[-1]

    #aff = aff.cpu()
    aff = output[2]
    frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)

    return frame2_seg

def recoginition(img_ref, img_tar, bbox_ref, bbox_tar, seg_ref, model):
    """
    - F_ref: feature of reference frame
    - F_tar: feature of target frame
    - bbox_ref: bboxes of reference frame
    - bbox_tar: bboxes of target frame
    - seg_ref: segmentation of reference frame
    """
    F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    seg_ref = seg_ref.squeeze()
    _, c, h, w = F_tar.size()
    seg_pred = torch.zeros(seg_ref.size())
    for cnt, br in bbox_ref.items():
        if not (cnt in bbox_tar):
            continue
        bt = bbox_tar[cnt]
        if(br is None or bt is None):
            continue
        seg_cnt = seg_ref[cnt]

        # feature of patch in the next frame
        F_tar_box = F_tar[:, :, bt.top:bt.bottom, bt.left:bt.right]
        F_ref_box = F_ref[:, :, br.top:br.bottom, br.left:br.right]
        F_tar_box_flat = F_tar_box.contiguous().view(c,-1)
        F_ref_box_flat = F_ref_box.contiguous().view(c,-1)

        # affinity between two patches
        aff = torch.mm(F_ref_box_flat.permute(1,0), F_tar_box_flat)
        aff = torch.nn.functional.softmax(aff * args.temp, dim=0)
        # transfer segmentation from patch1 to patch2
        seg_ref_box = seg_cnt[br.top:br.bottom, br.left:br.right]
        seg_ref_box = seg_ref_box.unsqueeze(0).unsqueeze(0)
        aff = aff.unsqueeze(0)

        seg_tar_box = transform_topk(aff,seg_ref_box.cuda(),k=args.topk,
                      h2=F_tar_box.size(2),w2=F_tar_box.size(3))

        seg_pred[cnt,bt.top:bt.bottom, bt.left:bt.right] = seg_tar_box
    return seg_pred

def disappear(seg,bbox_ref,bbox_tar=None):
    b,c,h,w = seg.size()
    for cnt in range(c):
        if(torch.sum(seg[:,cnt,:,:]) < 3 or (not (cnt in bbox_ref))):
            return True
        if(bbox_ref[cnt] is None):
            return True
        if(bbox_ref[cnt].right - bbox_ref[cnt].left < 3 or bbox_ref[cnt].bottom - bbox_ref[cnt].top < 3):
            return True

        if(bbox_tar is not None):
            if(cnt not in bbox_tar.keys()):
                return True
            if(bbox_tar[cnt] is None):
                return True
            if(bbox_tar[cnt].right - bbox_tar[cnt].left < 3 or bbox_tar[cnt].bottom - bbox_tar[cnt].top < 3):
                return True
    return False

def test(model, frame_list, video_dir, first_seg, large_seg, first_bbox):
    video_dir = os.path.join(video_dir)
    video_nm = video_dir.split('/')[-1]
    video_folder = os.path.join(args.out_dir, video_nm)
    os.makedirs(video_folder, exist_ok = True)
    os.makedirs(os.path.join(video_folder, 'track'), exist_ok = True)


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
            frame_tar_acc = frame_tar_acc.cpu()

            tmp_list = list(que.queue)
            if(len(tmp_list) > 0):
                pair = tmp_list[-1]
                framei = pair[0]
                segi = pair[1]
                bbox_pre = pair[2]
            else:
                bbox_pre = first_bbox
                framei = frame1
                segi = first_seg
            _, segi_int = torch.max(segi, dim=1)
            segi = to_one_hot(segi_int)
            bbox_tar, coords_ref_tar = bbox_next_frame(framei, segi, frame_tar, bbox_pre)

            
            if(bbox_tar is not None):
                if(1 in bbox_tar):
                    tmp = copy.deepcopy(bbox_tar[1])
                    if(tmp is not None):
                        tmp.upscale(8)
                        vis_bbox(frame_tar, tmp, os.path.join(video_folder, 'track', 'frame'+str(cnt+1)+'.png'), coords_ref_tar[1], segi[0,1,:,:])

            # previous 7 frames
            if not args.no_rec:
                tmp_queue = list(que.queue)
                for pair in tmp_queue:
                    framei = pair[0]
                    segi = pair[1]
                    bboxi = pair[2]
                    
                    if(bbox_tar is None or disappear(segi, bboxi, bbox_tar)):
                        frame_tar_est_i = forward(framei, frame_tar, model, segi)
                        frame_tar_est_i = frame_tar_est_i.cpu()
                    else:
                        frame_tar_est_i = recoginition(framei, frame_tar, bboxi, bbox_tar, segi, model)
                    
                    frame_tar_acc += frame_tar_est_i.cpu()
                frame_tar_avg = frame_tar_acc / (1 + len(tmp_queue))
            else:
                frame_tar_avg = frame_tar_acc

        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg",".png")
        out_path = os.path.join(video_folder,frame_nm)

        # upsampling & argmax
        frame_tar_up = torch.nn.functional.interpolate(frame_tar_avg,scale_factor=8,mode='bilinear')
        _, frame_tar_seg = torch.max(frame_tar_up.squeeze(), dim=0)

        frame_tar_seg = frame_tar_seg.squeeze().cpu().numpy()
        frame_tar_seg = np.array(frame_tar_seg, dtype=np.uint8)
        frame_tar_seg = scipy.misc.imresize(frame_tar_seg, (ori_h, ori_w), "nearest")
        imwrite_indexed(out_path,frame_tar_seg)

        if(not args.no_rec):
            if(que.qsize() == args.pre_num):
                que.get()
            seg = copy.deepcopy(frame_tar_avg.squeeze())
            frame, ori_h, ori_w = read_frame(frame_list[cnt], transforms)
            bbox_tar = seg2bbox_v2(frame_tar_up.cpu(), bbox_pre)
            que.put([frame,seg.unsqueeze(0),bbox_tar])

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
    small_seg = scipy.misc.imresize(seg, (tw//8,th//8),"nearest",mode="F")
    large_seg = scipy.misc.imresize(seg, (tw,th),"nearest",mode="F")

    t = []
    t.extend([transforms.ToTensor()])
    trans = transforms.Compose(t)
    pair = [large_seg, small_seg]
    transformed = list(trans(*pair))
    large_seg = transformed[0]
    small_seg = transformed[1]
    return to_one_hot(large_seg), to_one_hot(small_seg)

def draw_marker(im, win_num, coord):
    for II in range(win_num):
        for JJ in range(win_num):
            color = color_platte[(II-1)*win_num+JJ]
            color = (int(color[0]),int(color[1]),int(color[2]))
            for ii in range(II*win_size_h,(II+1)*win_size_h):
                for jj in range(JJ*win_size_w,(JJ+1)*win_size_w):
                    cv2.drawMarker(im,
                                  (int(coord[ii,jj,0]*8), int(coord[ii,jj,1]*8)),
                                  color,
                                  markerType=cv2.MARKER_CROSS,
                                  markerSize=3,
                                  thickness=1,
                                  line_type=cv2.LINE_AA)    

def vis_bbox(im, bbox, name, coords, seg):
    im = im * 128 + 128
    im = im.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
    fg_idx = seg.nonzero()


    for cnt in range(coords.size(0)):
        coord_i = coords[cnt]

        cv2.circle(im, (int(coord_i[0]*8), int(coord_i[1]*8)), 2, (0,255,0), thickness=-1)

    cv2.imwrite(name, im)



if(__name__ == '__main__'):
    args = parse_args()
    with open(args.val_txt) as f:
        lines = f.readlines()
    f.close()

    model = Model(pretrainRes=False, temp = args.temp, uselayer=4, model=args.model)
    if(args.multiGPU):
        model = nn.DataParallel(model)
    print("=> loading checkpoint '{}'".format(args.checkpoint_dir))
    checkpoint = torch.load(args.checkpoint_dir)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{} ({})' (epoch {})"
          .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
    torch.save(model.module.gray_encoder.state_dict(), 'track_switch_nopre_lc0_encoder.pth')
    model.cuda()
    model.eval()
    color_palette = np.loadtxt('libs/data/palette.txt',dtype=np.uint8).reshape(-1,3)



    cnt_list = [6,19]
    for cnt in range(0,30):
        
        line = lines[cnt]
        video_nm = line.strip()
        print('[{:n}/{:n}] Begin to segmentate video {}.'.format(cnt,len(lines),video_nm))

        video_dir = os.path.join(args.davis_dir, video_nm)
        frame_list = read_frame_list(video_dir)
        seg_dir = frame_list[0].replace("JPEGImages","Annotations")
        seg_dir = seg_dir.replace("jpg","png")
        large_seg, first_seg = read_seg(seg_dir)

        first_bbox = seg2bbox(large_seg, margin=0.6)
        for k,v in first_bbox.items():
            v.upscale(0.125)

        test(model, frame_list, video_dir, first_seg, large_seg, first_bbox)
        
        
