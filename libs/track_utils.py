import cv2
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.utils as vutils

from libs.vis_utils import norm_mask

############################# GLOBAL VARIABLES ########################
color_platte = [[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],
                [0, 0, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],
                [64, 0, 0],[192, 0, 0],[64, 128, 0],[192, 128, 0],[64, 0, 128],
                [192, 0, 128],[64, 128, 128],[192, 128, 128],[0, 64, 0],
                [128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128]]
color_platte = np.array(color_platte)

############################# HELPER FUNCTIONS ########################
class BBox():
    """
    bounding box class
    """
    def __init__(self, left, right, top, bottom, margin, h, w):
        if(margin > 0):
            bb_w = float(right - left)
            bb_h = float(bottom - top)
            margin_h = (bb_h * margin) / 2
            margin_w = (bb_w * margin) / 2
            left = left - margin_w
            right = right + margin_w
            top = top - margin_h
            bottom = bottom + margin_h
        self.left = max(math.floor(left), 0)
        self.right = min(math.ceil(right), w)
        self.top = max(math.floor(top), 0)
        self.bottom = min(math.ceil(bottom), h)

    def print(self):
        print("Left: {:n}, Right:{:n}, Top:{:n}, Bottom:{:n}".format(self.left, self.right, self.top, self.bottom))

    def upscale(self, scale):
        self.left   = math.floor(self.left * scale)
        self.right  = math.floor(self.right * scale)
        self.top    = math.floor(self.top * scale)
        self.bottom = math.floor(self.bottom * scale)

    def add(self, bbox):
        self.left += bbox.left
        self.right += bbox.right
        self.top += bbox.top
        self.bottom += bbox.bottom

    def div(self, num):
        self.left   /= num
        self.right  /= num
        self.top    /= num
        self.bottom /= num

def to_one_hot(y_tensor, n_dims=9):
    _,h,w = y_tensor.size()
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
    """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2,0,1).unsqueeze(0)

def create_grid(F_size, GPU=True):
    """
    INPUTS:
     - F_size: feature size
    OUTPUT:
     - return a standard grid coordinate
    """
    b, c, h, w = F_size
    theta = torch.tensor([[1,0,0],[0,1,0]])
    theta = theta.unsqueeze(0).repeat(b,1,1)
    theta = theta.float()

    # grid is a uniform grid with left top (-1,1) and right bottom (1,1)
    # b * (h*w) * 2
    grid = torch.nn.functional.affine_grid(theta, F_size)
    if(GPU):
        grid = grid.cuda()
    return grid

def seg2bbox_v2(seg, bbox_pre):
    """
    INPUTS:
     - seg: segmentation mask, a c*h*w one-hot tensor
    OUTPUS:
     - bbox: a c*4 tensor, indicating bbox for each object
    """
    seg = seg.squeeze()
    c,h,w = seg.size()
    bbox = {}
    bbox[0] = BBox(0, w, 0, h, 0, h, w)
    bbox[0].upscale(0.125)
    _, seg_int = torch.max(seg, dim=0)
    for cnt in range(1,c): # rule out background label
        seg_cnt = (seg_int == cnt) * 1
        # x * 2
        fg_idx = seg_cnt.nonzero().float()

        if(fg_idx.numel() > 0 and (bbox_pre[cnt] is not None)):
            fg_idx = torch.flip(fg_idx, (0,1))

            bbox_tmp = copy.deepcopy(bbox_pre[cnt])
            bbox_tmp.upscale(8)
            bbox[cnt] = coords2bbox_scale(fg_idx, h, w, bbox_tmp, margin=0.6, bandwidth=20)

            bbox[cnt].upscale(0.125)
        else:
            bbox[cnt] = None
    return bbox

def seg2bbox(seg, margin=0,print_info=False):
    """
    INPUTS:
     - seg: segmentation mask, a c*h*w one-hot tensor
    OUTPUS:
     - bbox: a c*4 tensor, indicating bbox for each object
    """
    seg = seg.squeeze()
    c,h,w = seg.size()
    bbox = {}
    bbox[0] = BBox(0, w, 0, h, 0, h, w)
    for cnt in range(1,c): # rule out background label
        seg_cnt = seg[cnt]
        # x * 2
        fg_idx = seg_cnt.nonzero()
        if(fg_idx.numel() > 0):
            left = fg_idx[:,1].min()
            right = fg_idx[:,1].max()
            top = fg_idx[:,0].min()
            bottom = fg_idx[:,0].max()
            bbox[cnt] = BBox(left, right, top, bottom, margin, h, w)
        else:
            bbox[cnt] = None
    return bbox

def gaussin(x, sigma):
    return torch.exp(- x ** 2 / (2 * (sigma ** 2)))

def calc_center(arr, mode='mean', sigma=10):
    """
    INPUTS:
     - arr: an array with coordinates, shape: n
     - mode: 'mean' to calculate Euclean center, 'mass' to calculate mass center
     - sigma: Gaussian parameter if calculating mass center
    """
    eu_center =  torch.mean(arr)
    if(mode == 'mean'):
        return eu_center
    # calculate weight center
    eu_center = eu_center.view(1,1).repeat(1,arr.size(0)).squeeze()
    diff = eu_center - arr
    weight = gaussin(diff, sigma)
    mass_center = torch.sum(weight * arr / (torch.sum(weight)))
    return mass_center

def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

def transform_topk(aff, frame1, k):
    """
    INPUTS:
     - aff: affinity matrix, b * N * N
     - frame1: reference frame
     - k: only aggregate top-k pixels with highest aff(j,i)
    """
    b,c,h,w = frame1.size()
    b, _, N = aff.size()
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim = 1, k=k)
    # b * N
    tk_val_min,_ = torch.min(tk_val, dim=1)
    tk_val_min = tk_val_min.view(b,1,N)
    aff[tk_val_min > aff] = 0
    aff = masked_softmax(aff, aff > 0, dim=1)
    frame1 = frame1.view(b,c,-1)
    frame2 = torch.bmm(frame1, aff)
    return frame2

def squeeze_all(*args):
    res = []
    for arg in args:
        res.append(arg.squeeze())
    return tuple(res)

def decode_seg(seg):
    seg = seg.squeeze()
    h,w = seg.size()
    color_seg = np.zeros((h,w,3))
    unis = np.unique(seg)
    seg = seg.numpy()
    for cnt,uni in enumerate(unis):
        xys = (seg == uni)
        xs,ys = np.nonzero(xys)
        color_seg[xs,ys,:] = color_platte[cnt].reshape(1,3)
    return color_seg

def draw_bbox(seg,bbox,color):
    """
    INPUTS:
     - segmentation, h * w * 3 numpy array
     - coord: left, right, top, bottom
    OUTPUT:
     - seg with a drawn bbox
    """
    seg = seg.copy()
    pt1 = (bbox.left,bbox.top)
    pt2 = (bbox.right,bbox.bottom)
    color = np.array(color, dtype=np.uint8)
    c = tuple(map(int, color))
    seg = cv2.rectangle(seg, pt1, pt2, c, 2)
    return seg

def vis_bbox(img, bbox, upscale=1):
    """
    INPUTS:
     - img: a h*w*c opencv image
     - bbox: a list of bounding box
    OUTPUT:
     - img: image with bbox drawn on
    """
    #for cnt in range(len(bbox)):
    for cnt, bbox_cnt in bbox.items():
        #bbox_cnt = bbox[cnt]
        if(bbox_cnt is None):
            continue
        bbox_cnt.upscale(upscale)
        color = color_platte[cnt+1]
        img = draw_bbox(img, bbox_cnt, color)
    return img

def vis_bbox_pair(bbox, bbox_next, frame1_var, frame2_var, out_path):
    frame1 = frame1_var * 128 + 128
    frame1 = frame1.squeeze().permute(1,2,0)
    im1 = cv2.cvtColor(np.array(frame1, dtype = np.uint8), cv2.COLOR_LAB2BGR)

    frame2 = frame2_var * 128 + 128
    frame2 = frame2.squeeze().permute(1,2,0)
    im2 = cv2.cvtColor(np.array(frame2, dtype = np.uint8), cv2.COLOR_LAB2BGR)

    #for i, (bbox_cnt, bbox_next_cnt) in enumerate(zip(bbox, bbox_next)):
    for cnt, bbox_cnt in bbox.items():
        bbox_next_cnt = bbox_next[cnt]
        bbox_cnt.upscale(8)
        bbox_next_cnt.upscale(8)
        im1 = draw_bbox(im1, bbox_cnt, color_platte[i+1])
        im2 = draw_bbox(im2, bbox_next_cnt, color_platte[i+1])

    im = np.concatenate((im1,im2), axis=1)
    cv2.imwrite(out_path, im)

def clean_seg(seg, bbox, threshold):
    c,h,w = seg.size()
    fgs = {}
    for cnt, bbox_cnt in bbox.items():
        if(bbox_cnt is not None):
            seg_cnt = seg[cnt]
            fg = seg_cnt.nonzero()
            fgs[cnt] = fg[:,[1,0]].float()
        else:
            fgs[cnt] = None
    fgs = clean_coords(fgs, bbox, threshold)
    seg_new = torch.zeros(seg.size())
    for cnt, fg in fgs.items():
        if(fg is not None):
            fg = fg.long()
            seg_new[cnt][fg[:,1],fg[:,0]] = seg[cnt][fg[:,1],fg[:,0]]
    return seg_new.view(1,c,h,w)

def clean_coords(coord, bbox_pre, threshold):
    """
    INPUT:
     - coord: coordinates of foreground points, a N * 2 tensor.
     - center: cluster center, a 2 tensor
     - threshold: we cut all points larger than this threshold
    METHOD: Rule out outliers in coord
    """
    new_coord = {}
    for cnt, coord_cnt in coord.items():
        bbox_pre_cnt = bbox_pre[cnt]
        if((bbox_pre_cnt is not None) and (coord_cnt is not None)):
            center_h = (bbox_pre_cnt.top  + bbox_pre_cnt.bottom)/2
            center_w = (bbox_pre_cnt.left + bbox_pre_cnt.right)/2
            dis = (coord_cnt[:,1] - center_h )**2 + (coord_cnt[:,0] - center_w)**2
            mean_ = torch.mean(dis)
            dis = dis / mean_

            idx_ = (dis <= threshold).nonzero()
            new_coord[cnt] = coord_cnt[idx_].squeeze()
        else:
            new_coord[cnt] = None
    return new_coord

def coord2bbox(bbox_pre, coord, h, w, adaptive=False):
    avg_h = calc_center(coord[:,1], mode='mass')
    avg_w = calc_center(coord[:,0], mode='mass')

    if adaptive:
        center = torch.mean(coord, dim = 0)
        dis_h = coord[:,1] - center[1]
        dis_w = coord[:,0] - center[0]

        dis_h = torch.mean(dis_h * dis_h, dim = 0)
        bb_height = (dis_h ** 0.5) * 8

        dis_w = torch.mean(dis_w * dis_w, dim = 0)
        bb_width = (dis_w ** 0.5) * 8

        # the adaptive method is sentitive to outliers, let's assume there's no dramatical change within
        # short range, so the height should not grow larger than 1.2 times height in previous frame
        bb_height = torch.min(bb_height, torch.Tensor([bbox_pre.bottom - bbox_pre.top]) * 1.2)
        bb_width = torch.min(bb_width, torch.Tensor([bbox_pre.right - bbox_pre.left]) * 1.2)

        bb_height = torch.max(bb_height, torch.Tensor([bbox_pre.bottom - bbox_pre.top]) * 0.8)
        bb_width = torch.max(bb_width, torch.Tensor([bbox_pre.right - bbox_pre.left]) * 0.8)

    else:
        bb_width  = float(bbox_pre.right  - bbox_pre.left)
        bb_height = float(bbox_pre.bottom - bbox_pre.top)
    left = avg_w - bb_width/2.0
    right = avg_w + bb_width/2.0
    top = avg_h - bb_height/2.0
    bottom = avg_h + bb_height/2.0

    coord_left = coord[:,0].min()
    coord_right = coord[:,0].max()
    coord_top = coord[:,1].min()
    coord_bottom = coord[:,1].max()

    bbox_tar_ = BBox(left   = int(max(left, 0)),
                     right  = int(min(right, w)),
                     top    = int(max(top, 0)),
                     bottom = int(min(bottom, h)),
                     margin = 0, h = h, w = w)
    return bbox_tar_

def post_process_seg(seg_pred):
    frame2_seg_bbox = torch.nn.functional.interpolate(seg_pred,scale_factor=8,mode='bilinear')
    frame2_seg_bbox = norm_mask(frame2_seg_bbox.squeeze())
    _, frame2_seg_bbox = torch.max(frame2_seg_bbox, dim=0)
    return frame2_seg_bbox

def post_process_bbox(seg, bbox_pre):
    fg_idx = seg.nonzero()

    bbox_pre_cnt = bbox_pre[cnt]
    if(bbox_pre_cnt is not None):
        bbox_pre_cnt.upscale(8)
        center_h = (bbox_pre_cnt.top  + bbox_pre_cnt.bottom) / 2
        center_w = (bbox_pre_cnt.left + bbox_pre_cnt.right) / 2
        fg_idx = clean_coord(fg_idx.float().cuda(), torch.Tensor([center_h, center_w]),keep_ratio=0.9)

def scatter_point(coord, name, w, h, center=None):
    fig, ax = plt.subplots()
    ax.axis((0,w,0,h))
    ax.scatter(coord[:,0], h - coord[:,1])
    ax.scatter(center[0], h - center[1], marker='^')

    plt.savefig(name)
    plt.clf()

def shift_bbox(seg, bbox):
    c,h,w = seg.size()
    bbox_new = {}
    for cnt in range(c):
        seg_cnt = seg[cnt]
        bbox_cnt = bbox[cnt]
        if(bbox_cnt is not None):
            fg_idx = seg_cnt.nonzero()
            fg_idx = fg_idx[:,[1,0]].float()
            center_h = calc_center(fg_idx[:,1], mode='mass')
            center_w = calc_center(fg_idx[:,0], mode='mass')

            # shift bbox w.r.t new center
            old_h = (bbox_cnt.top + bbox_cnt.bottom) / 2
            old_w = (bbox_cnt.left + bbox_cnt.right) / 2
            bb_width = bbox_cnt.right - bbox_cnt.left
            bb_height = bbox_cnt.bottom - bbox_cnt.top

            left = center_w - bb_width/2
            right = center_w + bb_width/2
            top = center_h - bb_height/2
            bottom = center_h + bb_height/2
            bbox_new[cnt] = BBox(left = left, right = right,
                                 top = top, bottom = bottom,
                                 margin = 0, h = h, w = w)
        else:
            bbox_new[cnt] = None
    return bbox_new

############################# MATCHING FUNCTIONS ########################
def match_ref_tar(F_ref, F_tar, seg_ref, temp):
    """
    INPUTS:
     - F_ref: feature of reference frame
     - F_tar: feature of target frame
     - seg_ref: segmentation of reference frame
     - temp: temperature of softmax
    METHOD:
     - take foreground pixels from the reference frame and match them to the
       target frame.
    RETURNS:
     - coord: a list of coordinates of foreground pixels in the target frame.
    """
    coords = {}
    c, h, w = F_ref.size()
    F_ref_flat = F_ref.view(c, -1)
    F_tar_flat = F_tar.view(c, -1)

    grid = create_grid(F_ref.unsqueeze(0).size()).squeeze()
    grid[:,:,0] = (grid[:,:,0]+1)/2 * w
    grid[:,:,1] = (grid[:,:,1]+1)/2 * h
    grid_flat = grid.view(-1,2)

    for cnt in range(seg_ref.size(0)):
        seg_cnt = seg_ref[cnt, :, :].view(-1)
        # there's no mask for this channel
        if(torch.sum(seg_cnt) < 2):
            coords[cnt] = None
            continue
        if(cnt > 0):
            fg_idx = seg_cnt.nonzero()
            F_ref_cnt_flat = F_ref_flat[:,fg_idx].squeeze()

        else:
            # for the background class, we just take the whole frame
            F_ref_cnt_flat = F_ref_flat
        aff = torch.mm(F_ref_cnt_flat.permute(1,0), F_tar_flat)
        aff = torch.nn.functional.softmax(aff*temp, dim = 1)
        coord = torch.mm(aff, grid_flat)
        coords[cnt] = coord
    return coords

def weighted_center(coords, center):
    """
    in_range = []
    for cnt in range(coords.shape[1]):
        coord_i = coords[0,cnt,:]
        if(np.linalg.norm(coord_i - prev_center) < bandwidth):
            in_range.append(coord_i)
    in_range = np.array(in_range)
    new_center = np.mean(in_range, axis=0)
    """
    center = center.reshape(1, 1, 2)

    dis_x = np.sqrt(np.power(coords[:,:,0] - center[:,:,0], 2))
    weight_x = 1 / dis_x
    weight_x = weight_x / np.sum(weight_x)
    dis_y = np.sqrt(np.power(coords[:,:,1] - center[:,:,1], 2))
    weight_y = 1 / dis_y
    weight_y = weight_y / np.sum(weight_y)

    new_x = np.sum(weight_x * coords[:,:,0])
    new_y = np.sum(weight_y * coords[:,:,1])

    return np.array([new_x, new_y]).reshape(1,1,2)

def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))

def neighbourhood_points(X, x_centroid, distance = 20):
    eligible_X = []
    #for x in X:
    for cnt in range(X.shape[0]):
        x = X[cnt,:]
        distance_between = euclid_distance(x, x_centroid)
        if distance_between <= distance:
            eligible_X.append(x)
    return eligible_X

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val

def mean_shift_center(coords, bandwidth=20):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame, 1xNx2
    """
    # 1 x 2
    avg_center = np.mean(coords, axis=0)
    prev_center = copy.deepcopy(avg_center)

    # pick the most close point as the center
    minimum = 100000
    for cnt in range(coords.shape[0]):
        coord_i = coords[cnt,:].reshape(1,2)
        dis = np.linalg.norm(coord_i - avg_center)
        if(dis < minimum):
            minimum = dis
            prev_center = copy.deepcopy(coord_i)

    counter = 0
    while True:
        counter += 1
        neighbors = neighbourhood_points(coords, prev_center.reshape(1,2), bandwidth)
        numerator = 0
        denominator = 0
        for neighbor in neighbors:
            distance = euclid_distance(neighbor, prev_center)
            weight = gaussian_kernel(distance, 4)
            numerator += (weight * neighbor)
            denominator += weight
        new_center = numerator / denominator

        if(np.sum(new_center - prev_center) < 0.1):
            final_center = torch.from_numpy(new_center)
            return final_center.view(1,2)
        else:
            prev_center = copy.deepcopy(new_center)

def coords2bbox_scale(coords, h_tar, w_tar, bbox_pre, margin, bandwidth, log=False):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame
     - h_tar: target image height
     - w_tar: target image widthg
    """
    b = 1
    center = mean_shift_center(coords.numpy(), bandwidth)

    center_repeat = center.repeat(coords.size(0),1)

    dis_x = torch.sqrt(torch.pow(coords[:,0] - center_repeat[:,0], 2))
    dis_x = torch.mean(dis_x, dim=0).detach()
    dis_y = torch.sqrt(torch.pow(coords[:,1] - center_repeat[:,1], 2))
    dis_y = torch.mean(dis_y, dim=0).detach()

    left = (center[:,0] - dis_x*2).view(b,1)
    right = (center[:,0] + dis_x*2).view(b,1)
    top = (center[:,1] - dis_y*2).view(b,1)
    bottom = (center[:,1] + dis_y*2).view(b,1)


    bbox_tar_ = BBox(left   = max(left, 0),
                     right  = min(right, w_tar),
                     top    = max(top, 0),
                     bottom = min(bottom, h_tar),
                     margin = margin, h = h_tar, w = w_tar)


    return bbox_tar_

def coord_wrt_bbox(coord, bbox):
    new_coord = []
    left = bbox.left
    right = bbox.right
    top = bbox.top
    bottom = bbox.bottom
    for cnt in range(coord.size(0)):
        coord_i = coord[cnt]
        if(coord_i[0] >= left and coord_i[0] <= right and coord_i[1] >= top and coord_i[1] <= bottom):
            new_coord.append(coord_i)
    new_coord = torch.stack(new_coord)
    return new_coord

def bbox_in_tar_scale(coords_tar, bbox_ref, h, w):
    """
    INPUTS:
     - coords_tar: foreground coordinates in the target frame
     - bbox_ref: bboxs in the reference frame
    METHOD:
     - calculate bbox in the next frame w.r.t pixels coordinates
    RETURNS:
     - each bbox in the tar frame
    """
    bbox_tar = {}
    #for cnt in range(len(bbox_ref)):
    for cnt, bbox_cnt in bbox_ref.items():
        if(cnt == 0):
            bbox_tar_ = BBox(left = 0,
                              right = w,
                              top = 0,
                              bottom = h,
                              margin = 0.6, h = h, w = w)
        elif(bbox_cnt is not None):
            if not (cnt in coords_tar):
                continue
            coord = coords_tar[cnt]
            if(coord is None):
                continue
            coord = coord.cpu()

            bbox_tar_ = coords2bbox_scale(coord, h, w, bbox_cnt, margin=1, bandwidth=5, log=(cnt==3))
        else:
            bbox_tar_ = None

        bbox_tar[cnt] = bbox_tar_
    return bbox_tar

############################# depreciated code, keep for safety ########################
"""
def bbox_in_tar(coords_tar, bbox_ref, h, w):
    INPUTS:
     - coords_tar: foreground coordinates in the target frame
     - bbox_ref: bboxs in the reference frame
    METHOD:
     - calculate bbox in the next frame w.r.t pixels coordinates
    RETURNS:
     - each bbox in the tar frame
    bbox_tar = {}
    for cnt, bbox_cnt in bbox_ref.items():
        if(cnt == 0):
            bbox_tar_ = BBox(left = 0,
                              right = w,
                              top = 0,
                              bottom = h,
                              margin = 0.5, h = h, w = w)
        elif(bbox_cnt is not None):
            if not (cnt in coords_tar):
                continue
            coord = coords_tar[cnt]
            center_h = (bbox_cnt.top + bbox_cnt.bottom) / 2
            center_w = (bbox_cnt.left + bbox_cnt.right) / 2
            if(coord is None):
                continue
            coord = coord.cpu()

            bbox_tar_ = coord2bbox(bbox_cnt, coord, h, w, adaptive=False)
        else:
            bbox_tar_ = None

        bbox_tar[cnt] = bbox_tar_
    return bbox_tar

def bbox_in_tar_v2(coords_tar, bbox_ref, h, w, seg_pre):
    INPUTS:
     - coords_tar: foreground coordinates in the target frame
     - bbox_ref: bboxs in the reference frame
    METHOD:
     - calculate bbox in the next frame w.r.t pixels coordinates
    RETURNS:
     - each bbox in the tar frame
    VERSION NOTE:
     - include scaling modeling
    bbox_tar = {}
    for cnt, bbox_cnt in bbox_ref.items():
        # for each channel
        if(cnt == 0):
            # for background channel
            bbox_tar_ = BBox(left = 0,
                              right = w,
                              top = 0,
                              bottom = h,
                              margin = 0.5, h = h, w = w)
        elif(bbox_cnt is not None):
            coord = coords_tar[cnt]
            if(coord is None):
                continue
            coord = coord.cpu()

            bbox_tar_ = coord2bbox(bbox_cnt, coord, h, w, seg_pre, adaptive=True)
        else:
            bbox_tar_ = None

        bbox_tar[cnt] = bbox_tar_
    return bbox_tar

def recoginition(F_ref, F_tar, bbox_ref, bbox_tar, seg_ref, temp):
    - F_ref: feature of reference frame
    - F_tar: feature of target frame
    - bbox_ref: bboxes of reference frame
    - bbox_tar: bboxes of target frame
    - seg_ref: segmentation of reference frame
    c, h, w = F_tar.size()
    seg_pred = torch.zeros(seg_ref.size())
    #for cnt,(br, bt) in enumerate(zip(bbox_ref, bbox_tar)):
    for cnt, br in bbox_ref.items():
        bt = bbox_tar[cnt]
        if(br is None or bt is None):
            continue
        seg_cnt = seg_ref[cnt]

        # feature of patch in the next frame
        F_tar_box = F_tar[:, bt.top:bt.bottom, bt.left:bt.right]
        F_ref_box = F_ref[:, br.top:br.bottom, br.left:br.right]
        F_tar_box_flat = F_tar_box.contiguous().view(c,-1)
        F_ref_box_flat = F_ref_box.contiguous().view(c,-1)

        # affinity between two patches
        aff = torch.mm(F_ref_box_flat.permute(1,0), F_tar_box_flat)
        aff = torch.nn.functional.softmax(aff * temp, dim=0)
        # transfer segmentation from patch1 to patch2
        seg_ref_box = seg_cnt[br.top:br.bottom, br.left:br.right]
        aff = aff.cpu()
        if(cnt == 0):
            seg_ref_box_flat = seg_ref_box.contiguous().view(-1)
            seg_tar_box = torch.mm(seg_ref_box_flat.unsqueeze(0), aff).squeeze()
        else:
        seg_ref_box_flat = seg_ref_box.contiguous().view(-1)
        seg_tar_box = torch.mm(seg_ref_box_flat.unsqueeze(0), aff).squeeze()
        #seg_tar_box = transform_topk(aff.unsqueeze(0), seg_ref_box.contiguous().unsqueeze(0).unsqueeze(0), 20)
        seg_tar_box = seg_tar_box.view(F_tar_box.size(1), F_tar_box.size(2))

        seg_pred[cnt,bt.top:bt.bottom, bt.left:bt.right] = seg_tar_box
    return seg_pred

def bbox_next_frame_v2(F_first, F_pre, seg_pre, seg_first, F_tar, bbox_first, bbox_pre, temp, direct=False):
    INPUTS:
     - direct: rec|direct,
       - if False, use previous frame to locate bbox
       - if True, use first frame to locate bbox
    F_first, F_pre, seg_pre, seg_first, F_tar = squeeze_all(F_first, F_pre, seg_pre, seg_first, F_tar)
    c, h, w = F_first.size()
    if not direct:
        coords_tar = match_ref_tar(F_pre, F_tar, seg_pre, temp)
    else:
        coords_tar = match_ref_tar(F_first, F_tar, seg_first, temp)

    bbox_tar = bbox_in_tar(coords_tar, bbox_first, h, w)

    seg_pred = recoginition(F_first, F_tar, bbox_first, bbox_tar, seg_first, temp)
    return seg_pred.unsqueeze(0)

def bbox_next_frame_v3(F_first, F_pre, seg_pre, seg_first, F_tar, bbox_first, bbox_pre, temp, name):
    METHOD: combining tracking & direct recognition, calculate bbox in target frame
            using both first frame and previous frame.
    F_first, F_pre, seg_pre, seg_first, F_tar = squeeze_all(F_first, F_pre, seg_pre, seg_first, F_tar)
    c, h, w = F_first.size()

    coords_pre_tar = match_ref_tar(F_pre, F_tar, seg_pre, temp)
    coords_first_tar = match_ref_tar(F_first, F_tar, seg_first, temp)
    coords_tar = {}
    for cnt, coord_first in coords_first_tar.items():
        coord_pre = coords_pre_tar[cnt]
        # fall-back schema
        if(coord_pre is None):
            coord_tar_ = coord_first
        else:
            coord_tar_ = coord_pre
        coords_tar[cnt] = coord_tar_
    _, seg_pre_idx = torch.max(seg_pre, dim = 0)

    coords_tar = clean_coords(coords_tar, bbox_pre, threshold=4)
    bbox_tar = bbox_in_tar(coords_tar, bbox_first, h, w)

    # recoginition
    seg_pred = recoginition(F_first, F_tar, bbox_first, bbox_tar, seg_first, temp)
    seg_cleaned = clean_seg(seg_pred, bbox_tar, threshold=1)

    # move bbox w.r.t cleaned seg
    bbox_tar = shift_bbox(seg_cleaned, bbox_tar)

    seg_post = post_process_seg(seg_pred.unsqueeze(0))
    return seg_pred, seg_post, bbox_tar

def bbox_next_frame_v4(F_first, F_pre, seg_pre, seg_first, F_tar, bbox_first,
                       bbox_pre, temp):
    METHOD: combining tracking & direct recognition, calculate bbox in target frame
            using both first frame and previous frame.
    Version Note: include bounding box scaling
    F_first, F_pre, seg_pre, seg_first, F_tar = squeeze_all(F_first, F_pre, seg_pre, seg_first, F_tar)
    c, h, w = F_first.size()

    coords_pre_tar = match_ref_tar(F_pre, F_tar, seg_pre, temp)
    coords_first_tar = match_ref_tar(F_first, F_tar, seg_first, temp)
    coords_tar = {}
    for cnt, coord_first in coords_first_tar.items():
        coord_pre = coords_pre_tar[cnt]
        # fall-back schema
        if(coord_pre is None):
            coord_tar_ = coord_first
        else:
            coord_tar_ = coord_pre
        coords_tar[cnt] = coord_tar_
    _, seg_pre_idx = torch.max(seg_pre, dim = 0)

    coords_tar = clean_coords(coords_tar, bbox_pre, threshold=4)

    bbox_tar = bbox_in_tar_v2(coords_tar, bbox_first, h, w, seg_pre)

    # recoginition
    seg_pred = recoginition(F_first, F_tar, bbox_first, bbox_tar, seg_first, temp)
    seg_cleaned = clean_seg(seg_pred, bbox_tar, threshold=1)

    # move bbox w.r.t cleaned seg
    bbox_tar = shift_bbox(seg_cleaned, bbox_tar)

    seg_post = post_process_seg(seg_pred.unsqueeze(0))
    return seg_pred, seg_post, bbox_tar

"""
"""
def bbox_next_frame(F_ref, seg_ref, F_tar, bbox, temp):
    # b * h * w * 2
    b, c, h, w = F_ref.size()
    grid = create_grid(F_ref.size()).squeeze()
    grid[:,:,0] = (grid[:,:,0]+1)/2 * w
    grid[:,:,1] = (grid[:,:,1]+1)/2 * h
    # grid_flat: (h * w) * 2
    grid_flat = grid.view(-1,2)
    seg_ref = seg_ref.squeeze()
    F_ref = F_ref.squeeze()
    F_ref_flat = F_ref.view(c,-1)
    F_tar = F_tar.squeeze()
    F_tar_flat = F_tar.view(c,-1)
    bbox_next = []
    seg_pred = torch.zeros(seg_ref.size())
    for i in range(0,seg_ref.size(0)):
        seg_cnt = seg_ref[i,:,:].contiguous().view(-1)
        if(seg_cnt.max() == 0):
            continue
        bbox_cnt = bbox[i]
        if(i > 0):
            fg_idx = seg_cnt.nonzero()
            # take pixels of this instance out
            F_ref_cnt_flat = F_ref_flat[:,fg_idx].squeeze()

            # affinity between patch and target frame
            # aff: (hh * ww, h * w)
            aff = torch.mm(F_ref_cnt_flat.permute(1,0), F_tar_flat)
            aff = torch.nn.functional.softmax(aff*temp, dim = 1)
            # coord of this patch in next frame: (hh*ww) * 2
            coord = torch.mm(aff, grid_flat)
            avg_h = calc_center(coord[:,1], mode='mass').cpu().long()
            avg_w = calc_center(coord[:,0], mode='mass').cpu().long()
            bb_width = bbox_cnt.right - bbox_cnt.left
            bb_height = bbox_cnt.bottom - bbox_cnt.top
            bbox_next_ = BBox(left = max(avg_w - bb_width/2,0),
                                  right = min(avg_w + bb_width/2, w),
                                  top = max(avg_h - bb_height/2,0),
                                  bottom = min(avg_h + bb_height/2,h),
                                  margin = 0, h = h, w = w)
        else:
            bbox_next_ = BBox(left = 0,
                              right = w,
                              top = 0,
                              bottom = h,
                              margin = 0, h = h, w = w)
        bbox_next.append(bbox_next_)

        # feature of patch in the next frame
        F_tar_box = F_tar[:, bbox_next_.top:bbox_next_.bottom, bbox_next_.left:bbox_next_.right]
        F_ref_box = F_ref[:, bbox_cnt.top:bbox_cnt.bottom, bbox_cnt.left:bbox_cnt.right]
        F_tar_box_flat = F_tar_box.contiguous().view(c,-1)
        F_ref_box_flat = F_ref_box.contiguous().view(c,-1)

        # affinity between two patches
        aff = torch.mm(F_ref_box_flat.permute(1,0), F_tar_box_flat)
        aff = torch.nn.functional.softmax(aff * temp, dim=0)
        # transfer segmentation from patch1 to patch2
        seg_ref_box = seg_ref[i, bbox_cnt.top:bbox_cnt.bottom, bbox_cnt.left:bbox_cnt.right]
        aff = aff.cpu()
        seg_ref_box_flat = seg_ref_box.contiguous().view(-1)
        seg_tar_box = torch.mm(seg_ref_box_flat.unsqueeze(0), aff).squeeze()
        seg_tar_box = seg_tar_box.view(F_tar_box.size(1),F_tar_box.size(2))

        seg_pred[i,bbox_next_.top:bbox_next_.bottom, bbox_next_.left:bbox_next_.right] = seg_tar_box

    #seg_pred[0,:,:] = 1 - torch.sum(seg_pred[1:,:,:],dim=0)

    return seg_pred.unsqueeze(0), bbox_next
def bbox_next_frame_rec(F_first, F_ref, seg_ref, seg_first,
                        F_tar, bbox_first, bbox, temp):
    # b * h * w * 2
    b, c, h, w = F_ref.size()
    seg_ref = seg_ref.squeeze()
    F_ref = F_ref.squeeze()
    F_ref_flat = F_ref.view(c,-1)
    F_tar = F_tar.squeeze()
    F_tar_flat = F_tar.view(c,-1)
    F_first = F_first.squeeze()
    bbox_next = []
    seg_pred = torch.zeros(seg_ref.size())
    seg_first = seg_first.squeeze()
    for i in range(0,seg_ref.size(0)):
        seg_cnt = seg_ref[i,:,:].contiguous().view(-1)
        if(seg_cnt.max() == 0):
            continue
        if(i > len(bbox)-1):
            continue
        bbox_cnt = bbox[i]
        bbox_first_cnt = bbox_first[i]
        if(i > 0):
            fg_idx = seg_cnt.nonzero()
            F_ref_cnt_flat = F_ref_flat[:,fg_idx].squeeze()

            # affinity between patch and target frame
            if(F_ref_cnt_flat.dim() < 2):
                # some small objects may miss
                continue
            aff = torch.mm(F_ref_cnt_flat.permute(1,0), F_tar_flat)
            aff = torch.nn.functional.softmax(aff*temp, dim = 1)
            coord = torch.mm(aff, grid_flat)
            #coord = transform_topk(aff.unsqueeze(0), grid.unsqueeze(0), dim=2, k=20)
            avg_h = calc_center(coord[:,1], mode='mass').cpu()
            avg_w = calc_center(coord[:,0], mode='mass').cpu()
            bb_width  = float(bbox_first_cnt.right  - bbox_first_cnt.left)
            bb_height = float(bbox_first_cnt.bottom - bbox_first_cnt.top)
            coord = coord.cpu()

            left = avg_w - bb_width/2.0
            right = avg_w + bb_width/2.0
            top = avg_h - bb_height/2.0
            bottom = avg_h + bb_height/2.0

            bbox_next_ = BBox(left   = int(max(left, 0)),
                              right  = int(min(right, w)),
                              top    = int(max(top, 0)),
                              bottom = int(min(bottom, h)),
                              margin = 0, h = h, w = w)
        else:
            bbox_next_ = BBox(left = 0,
                              right = w,
                              top = 0,
                              bottom = h,
                              margin = 0, h = h, w = w)

        bbox_next.append(bbox_next_)

        # feature of patch in the next frame
        F_tar_box = F_tar[:, bbox_next_.top:bbox_next_.bottom, bbox_next_.left:bbox_next_.right]
        F_ref_box = F_first[:, bbox_first_cnt.top:bbox_first_cnt.bottom, bbox_first_cnt.left:bbox_first_cnt.right]
        F_tar_box_flat = F_tar_box.contiguous().view(c,-1)
        F_ref_box_flat = F_ref_box.contiguous().view(c,-1)
    print('================')

        # affinity between two patches
        aff = torch.mm(F_ref_box_flat.permute(1,0), F_tar_box_flat)
        aff = torch.nn.functional.softmax(aff * temp, dim=0)
        # transfer segmentation from patch1 to patch2
        seg_ref_box = seg_first[i, bbox_first_cnt.top:bbox_first_cnt.bottom, bbox_first_cnt.left:bbox_first_cnt.right]
        aff = aff.cpu()
        seg_ref_box_flat = seg_ref_box.contiguous().view(-1)
        seg_tar_box = torch.mm(seg_ref_box_flat.unsqueeze(0), aff).squeeze()
        #seg_tar_box = transform_topk(aff.unsqueeze(0), seg_ref_box.contiguous().unsqueeze(0).unsqueeze(0))
        seg_tar_box = seg_tar_box.view(F_tar_box.size(1),F_tar_box.size(2))

        seg_pred[i,bbox_next_.top:bbox_next_.bottom, bbox_next_.left:bbox_next_.right] = seg_tar_box

    return seg_pred.unsqueeze(0), bbox_next

def transform_topk(aff, frame1, k=20, dim=1):
    b,c,h,w = frame1.size()
    b, N1, N2 = aff.size()
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim = dim, k=k)
    # b * N
    tk_val_min,_ = torch.min(tk_val,dim=dim)
    if(dim == 1):
        tk_val_min = tk_val_min.view(b,1,N2)
    else:
        tk_val_min = tk_val_min.view(b,N1,1)
    aff[tk_val_min > aff] = 0
    frame1 = frame1.view(b,c,-1)
    if(dim == 1):
        frame2 = torch.bmm(frame1, aff)
        return frame2
    else:
        frame2 = torch.bmm(aff, frame1.permute(0,2,1))
        return frame2.squeeze()
"""
