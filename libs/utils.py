import os
import cv2
import glob
import torch
import scipy.misc
import numpy as np
from PIL import Image
import libs.transforms_multi as transforms

def print_options(opt, test=False):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if(test):
        expr_dir = os.path.join(opt.out_dir)
        os.makedirs(expr_dir,exist_ok=True)
        file_name = os.path.join(expr_dir, 'test_opt.txt')
    else:
        expr_dir = os.path.join(opt.checkpoint_dir)
        os.makedirs(expr_dir,exist_ok=True)
        file_name = os.path.join(expr_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list

def read_frame(frame_dir, transforms):
    frame = cv2.imread(frame_dir)
    ori_h,ori_w,_ = frame.shape
    if(ori_h > ori_w):
    	tw = ori_w
    	th = (tw * ori_h) / ori_w
    	th = int((th // 64) * 64)
    else:
    	th = ori_h
    	tw = (th * ori_w) / ori_h
    	tw = int((tw // 64) * 64)
    #h = (ori_h // 64) * 64
    #w = (ori_w // 64) * 64
    frame = cv2.resize(frame, (tw,th))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    pair = [frame, frame]
    transformed = list(transforms(*pair))
    return transformed[0].cuda().unsqueeze(0), ori_h, ori_w

def to_one_hot(y_tensor, n_dims=9):
    _,h,w = y_tensor.size()
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2,0,1).unsqueeze(0)

def read_seg(seg_dir, crop_size):
    seg = Image.open(seg_dir)
    h,w = seg.size
    if(h > w):
    	tw = crop_size
    	th = (tw * h) / w
    	th = int((th // 64) * 64)
    else:
    	th = crop_size
    	tw = (th * w) / h
    	tw = int((tw // 64) * 64)
    seg = np.asarray(seg).reshape((w,h,1))
    seg = np.squeeze(seg)
    seg = scipy.misc.imresize(seg, (tw//8,th//8),"nearest",mode="F")

    seg = torch.from_numpy(seg).view(1,tw//8,th//8)
    return to_one_hot(seg)

def create_transforms(crop_size):
    normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
    t = []
    t.extend([transforms.ToTensor(),
    		  normalize])
    return transforms.Compose(t)

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

def norm_mask(mask):
    """
    INPUTS:
     - mask: segmentation mask
    """
    c,h,w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def diff_crop(F, x1, y1, x2, y2, ph, pw):
    """
    Differatiable cropping
    INPUTS:
     - F: frame feature
     - x1,y1,x2,y2: top left and bottom right points of the patch
     - theta is defined as :
                        a b c
                        d e f
    """
    bs, ch, h, w = F.size()
    a = ((x2-x1)/w).view(bs,1,1)
    b = torch.zeros(a.size()).cuda()
    c = (-1+(x1+x2)/w).view(bs,1,1)
    d = torch.zeros(a.size()).cuda()
    e = ((y2-y1)/h).view(bs,1,1)
    f = (-1+(y2+y1)/h).view(bs,1,1)
    theta_row1 = torch.cat((a,b,c),dim=2)
    theta_row2 = torch.cat((d,e,f),dim=2)
    theta = torch.cat((theta_row1, theta_row2),dim=1).cuda()
    size = torch.Size((bs,ch,pw,ph))
    grid = torch.nn.functional.affine_grid(theta, size)
    patch = torch.nn.functional.grid_sample(F,grid)
    return patch

def center2bbox(center, patch_size, h, w):
    b = center.size(0)
    if(isinstance(patch_size,int)):
        new_l = center[:,0] - patch_size/2
    else:
        new_l = center[:,0] - patch_size[1]/2
    new_l[new_l < 0] = 0
    new_l = new_l.view(b,1)

    if(isinstance(patch_size,int)):
        new_r = new_l + patch_size
    else:
        new_r = new_l + patch_size[1]
    new_r[new_r > w] = w

    if(isinstance(patch_size,int)):
        new_t = center[:,1] - patch_size/2
    else:
        new_t = center[:,1] - patch_size[0]/2
    new_t[new_t < 0] = 0
    new_t = new_t.view(b,1)

    if(isinstance(patch_size,int)):
        new_b = new_t + patch_size
    else:
        new_b = new_t + patch_size[0]
    new_b[new_b > h] = h

    new_center = torch.cat((new_l,new_r,new_t,new_b),dim=1)
    return new_center
