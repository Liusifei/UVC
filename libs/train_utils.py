import os
import cv2
import torch
import shutil
# import visdom
import numpy as np
from libs.vis_utils import draw_certainty_map, flow_to_rgb, prepare_img
from os.path import join


def draw_bbox(img,bbox):
	"""
	INPUTS:
	 - segmentation, h * w * 3 numpy array
	 - bbox: left, right, top, bottom
	OUTPUT:
	 - image with a drawn bbox
	"""
	# print("bbox: ", bbox)
	pt1 = (int(bbox[0]),int(bbox[2]))
	pt2 = (int(bbox[1]),int(bbox[3]))
	color = np.array([51,255,255], dtype=np.uint8)
	c = tuple(map(int, color))
	img = cv2.rectangle(img, pt1, pt2, c, 5)
	return img


def save_vis(pred2, gt2, frame1, frame2, savedir, new_c=None):
	"""
	INPUTS:
	 - pred: predicted patch, a 3xpatch_sizexpatch_size tensor
	 - gt2: GT patch, a 3xhxw tensor
	 - gt1: first GT frame, a 3xhxw tensor
	 - gt_grey: whether to use ground trught L channel in predicted image
	"""
	b = pred2.size(0)
	pred2 = pred2 * 128 + 128
	gt2 = gt2 * 128 + 128
	frame1 = frame1 * 128 + 128
	frame2 = frame2 * 128 + 128


	for cnt in range(b):
		im = pred2[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_bgr = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)
		im_pred = np.clip(im_bgr, 0, 255)

		im = gt2[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_gt2 = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

		im = frame1[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_frame1 = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

		im = frame2[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_frame2 = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)
		
		if new_c is not None:
			new_bbox = new_c[cnt]
			im_frame2 = draw_bbox(im_frame2,new_bbox)
			im_frame2 = cv2.resize(im_frame2, (im_frame1.shape[0],im_frame1.shape[1]))
			
			im = np.concatenate((im_frame1, im_frame2), axis = 1)
			cv2.imwrite(os.path.join(savedir, "{:02d}_loc.png".format(cnt)), im)

		im = np.concatenate((im_frame1, im_pred, im_gt2), axis = 1)
		cv2.imwrite(os.path.join(savedir, "{:02d}_patch.png".format(cnt)), im)
		

def save_vis_ae(pred, gt, savepath):
	b = pred.size(0)
	for cnt in range(b):
		im = pred[cnt].cpu().detach() * 128 + 128
		im = im.numpy().transpose(1,2,0)
		im_pred = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)
		#im_pred = np.clip(im_bgr, 0, 255)

		im = gt[cnt].cpu().detach() * 128 + 128
		im = im.numpy().transpose(1,2,0)
		im_gt = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

		im = np.concatenate((im_gt, im_pred), axis = 1)
		cv2.imwrite(os.path.join(savepath, "{:02d}.png".format(cnt)), im)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", savedir="models"):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, os.path.join(savedir, 'model_best.pth.tar'))

def sample_patch(b,h,w,patch_size):
	left = randint(0, max(w - patch_size,1))
	top  = randint(0, max(h - patch_size,1))
	right = left + patch_size
	bottom = top + patch_size
	return torch.Tensor([left, right, top, bottom]).view(1,4).repeat(b,1).cuda()
    

def log_current(epoch, loss_ave, best_loss, filename = "log_current.txt", savedir="models"):
    file = join(savedir, filename)
    with open(file, "a") as text_file:
        print("epoch: {}".format(epoch), file=text_file)
        print("best_loss: {}".format(best_loss), file=text_file)
        print("current_loss: {}".format(loss_ave), file=text_file)