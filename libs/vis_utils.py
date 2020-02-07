import cv2
import torch
import numpy as np
import seaborn as sns
from model import transform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def prepare_img(img):

    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR

    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    #cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return img

def flow_to_rgb(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.float32(img) / 255.0


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def aff2flow(A, F_size, GPU=True):
    """
    INPUT:
     - A: a (H*W)*(H*W) affinity matrix
     - F_size: image feature size
    OUTPUT:
     - U: a (2*H*W) flow tensor, U_ij indicates the coordinates of pixel ij in target image.
    """
    b,c,h,w = F_size
    theta = torch.tensor([[1,0,0],[0,1,0]])
    theta = theta.unsqueeze(0).repeat(b,1,1)
    theta = theta.float()

    # grid is a uniform grid with left top (-1,1) and right bottom (1,1)
    # b * (h*w) * 2
    grid = torch.nn.functional.affine_grid(theta, F_size)
    #grid = grid.view(b,h*w,2)
    if(GPU):
        grid = grid.cuda()
    # b * (h*w) * 2
    # A: 1x1024x1024
    grid = grid.permute(0,3,1,2)
    U = transform(A, grid)
    return (U - grid).permute(0,2,3,1)

def draw_certainty_map(map, normalize=False):
    """
    INPUTS:
     - map: certainty map of flow
    """
    map = map.squeeze()
    # normalization
    if(normalize):
        map = (map - map.min())/(map.max() - map.min())
        # draw heat map
        ax = sns.heatmap(map, yticklabels=False, xticklabels=False, cbar=True)
    else:
        ax = sns.heatmap(map, yticklabels=False, xticklabels=False, vmin=0.0, vmax=1.0, cbar=True)
    figure = ax.get_figure()
    width, height = figure.get_size_inches() * figure.get_dpi()

    canvas = FigureCanvas(figure)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    # crop border
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray[gray == 255] = 0
    gray[gray > 0] = 255
    coords = cv2.findNonZero(gray)
    x,y,w,h = cv2.boundingRect(coords)
    #rect = image[y:y+h, x:x+w]
    rect = image
    figure.clf()
    return rect
