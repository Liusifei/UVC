# OS libraries
import os
import cv2
import glob
import scipy.misc
import numpy as np
from PIL import Image

# Pytorch
import torch

# Customized libraries
import libs.transforms_pair as transforms

color_palette = np.loadtxt('libs/data/palette.txt',dtype=np.uint8).reshape(-1,3)

def transform_topk(aff, frame1, k, h2=None, w2=None):
	"""
	INPUTS:
	 - aff: affinity matrix, b * N * N
	 - frame1: reference frame
	 - k: only aggregate top-k pixels with highest aff(j,i)
	 - h2, w2, frame2's height & width
	OUTPUT:
	 - frame2: propagated mask from frame1 to the next frame
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

def read_frame(frame_dir, transforms, scale_size):
    """
    read a single frame & preprocess
    """
    frame = cv2.imread(frame_dir)
    ori_h,ori_w,_ = frame.shape
    # scale, makes height & width multiples of 64
    if(len(scale_size) == 1):
        if(ori_h > ori_w):
        	tw = scale_size[0]
        	th = (tw * ori_h) / ori_w
        	th = int((th // 64) * 64)
        else:
        	th = scale_size[0]
        	tw = (th * ori_w) / ori_h
        	tw = int((tw // 64) * 64)
    else:
        tw = scale_size[1]
        th = scale_size[0]
    frame = cv2.resize(frame, (tw,th))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    pair = [frame, frame]
    transformed = list(transforms(*pair))
    return transformed[0].cuda().unsqueeze(0), ori_h, ori_w

def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2,0,1).unsqueeze(0)

def read_seg(seg_dir, scale_size):
    seg = Image.open(seg_dir)
    h,w = seg.size
    if(len(scale_size) == 1):
    	if(h > w):
    		tw = scale_size[0]
    		th = (tw * h) / w
    		th = int((th // 64) * 64)
    	else:
    		th = scale_size[0]
    		tw = (th * w) / h
    		tw = int((tw // 64) * 64)
    else:
    	tw = scale_size[1]
    	th = scale_size[0]
    seg = np.asarray(seg).reshape((w,h,1))
    seg_ori = np.squeeze(seg)
    small_seg = scipy.misc.imresize(seg_ori, (tw//8,th//8),"nearest",mode="F")
    large_seg = scipy.misc.imresize(seg_ori, (tw,th),"nearest",mode="F")

    t = []
    t.extend([transforms.ToTensor()])
    trans = transforms.Compose(t)
    pair = [large_seg, small_seg]
    transformed = list(trans(*pair))
    large_seg = transformed[0]
    small_seg = transformed[1]
    return to_one_hot(large_seg), to_one_hot(small_seg), seg_ori

def imwrite_indexed(filename,array):
  """ Save indexed png for DAVIS."""
  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')
