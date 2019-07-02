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
		# ori_bbox = ori_c[cnt]
		# im_frame1 = draw_bbox(im_frame1,ori_bbox)

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
	grid = FUNC.affine_grid(theta, size)
	patch = FUNC.grid_sample(F,grid)
	return patch

def log_current(epoch, loss_ave, best_loss, filename = "log_current.txt", savedir="models"):
    file = join(savedir, filename)
    with open(file, "a") as text_file:
        print("epoch: {}".format(epoch), file=text_file)
        print("best_loss: {}".format(best_loss), file=text_file)
        print("current_loss: {}".format(loss_ave), file=text_file)
# class loss_plotter():
#     def __init__(self, port = 8097, server = "http://localhost"):
#         self.vis = visdom.Visdom(port=port, server=server)
#         assert self.vis.check_connection(timeout_seconds=3),'No connection could be formed quickly'
#         self.wins = {}
#         self.losses = {}
#         self.cnt = 0

#     def plot(self, losses, names):
#         self.cnt += 1
#         X = np.array(range(1, self.cnt+1))
#         for name,loss in zip(names, losses):
#             if not (name in self.wins):
#                 self.losses[name] = []
#                 self.losses[name].append(loss)
#                 Y = np.array(self.losses[name])
#                 self.wins[name] = self.vis.line(
#                     Y = Y,
#                     X = X,
#                     opts = dict(markers=False, legend=[name])
#                 )
#             else:
#                 self.losses[name].append(loss)
#                 Y = np.array(self.losses[name])
#                 self.vis.line(
#                     Y = Y,
#                     X = X,
#                     opts = dict(markers=False, legend=[name]),
#                     win = self.wins[name]
#                 )
