import os
import numpy as np
from PIL import Image
n_cl = 20
CLASSES = ['background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
           'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
           'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe', 'left-shoe']
#GT_DIR = '/scratch/xiaolonw/VIP/Category_ids/'
#PRE_DIR = '/scratch/xiaolonw/VIP/SegmentationPred_pastk/'

# PRE_DIR = '/scratch/xiaolonw/VIP/results/'
#PRE_DIR = '/scratch/xiaolonw/VIP_results_mask_kin+vlog/results/'

GT_DIR = '/media/xtli/eb0943df-a3fc-4ae2-a6e5-021cfdcfec3d/home/xtli/DATA/VIP/VIP_Fine/Annotations/Category_ids/'
PRE_DIR = '/home/xtli/Dropbox/Neurips2019/results/VIP/category//'

def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)

def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    index_map = {}
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7-j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7-j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7-j))
            cid = np.right_shift(cid, 3)

        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
        index_map['%d_%d_%d'%(r, g, b)] = i
    return color_map, index_map

#
# def init_path():
#     image_dir = PRE_DIR
#     label_dir = GT_DIR
#
#     file_names = []
#     for vid in os.listdir(image_dir):
#         for img in os.listdir(os.path.join(image_dir, vid, 'gray')):
#             j = img.find('_')
#             if img[:j] == 'global':
#                 file_names.append([vid, img[j+1:-4]])
#     print ("result of", image_dir)
#
#     image_paths = []
#     label_paths = []
#     for file_name in file_names:
#         image_paths.append(os.path.join(image_dir, file_name[0], 'gray', 'global_'+file_name[1]+'.png'))
#         label_paths.append(os.path.join(label_dir, file_name[0], file_name[1]+'.png'))
#     return image_paths, label_paths


def init_path():
    image_dir = PRE_DIR
    label_dir = GT_DIR

    file_names = []
    for vid in os.listdir(image_dir):
        for img in os.listdir(os.path.join(image_dir, vid)):
            file_names.append([vid, img])
    print ("result of", image_dir)

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name[0], file_name[1]))
        label_paths.append(os.path.join(label_dir, file_name[0], file_name[1]))
    return image_paths, label_paths


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(images, labels):

    color_map, index_map = _get_voc_color_map()
    hist = np.zeros((n_cl, n_cl))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape

        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cl)

    return hist

def show_result(hist):

    classes = CLASSES
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print ('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print ('>>>', 'overall accuracy', acc)
    print ('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print ('Accuracy for each class (pixel accuracy):')
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print ('>>>', 'mean accuracy', np.nanmean(acc))
    print ('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print ('>>>', 'mean IU', np.nanmean(iu))
    print ('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print ('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print ('=' * 50)



if __name__ == '__main__':
    main()
