import os
import cv2
import torch
import shutil
# import visdom
import numpy as np
from os.path import join

def save_vis(pred, gt2, gt1, out_dir, gt_grey=False, prefix=0):
	"""
	INPUTS:
	 - pred: predicted Lab image, a 3xhxw tensor
	 - gt2: second GT frame, a 3xhxw tensor
	 - gt1: first GT frame, a 3xhxw tensor
	 - out_dir: output image save path
	 - gt_grey: whether to use ground trught L channel in predicted image
	"""
	b = pred.size(0)
	pred = pred * 128 + 128
	gt1 = gt1 * 128 + 128
	gt2 = gt2 * 128 + 128

	if(gt_grey):
		pred[:,0,:,:] = gt2[:,0,:,:]
	for cnt in range(b):
		im = pred[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_bgr = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)
		im_pred = np.clip(im_bgr, 0, 255)

		im = gt2[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_gt2 = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

		im = gt1[cnt].cpu().detach().numpy().transpose( 1, 2, 0)
		im_gt1 = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

		im = np.concatenate((im_gt1, im_gt2, im_pred), axis = 1)
		print(out_dir, "{:02d}{:02d}.png".format(prefix, cnt))
		cv2.imwrite(join(out_dir, "{:02d}{:02d}.png".format(prefix, cnt)), im)

def save_vis_ae(pred, gt, savepath):
	b = pred.size(0)
	for cnt in range(b):
		im = pred[cnt].cpu().detach() * 128 + 128
		im = im.numpy().transpose(1,2,0)
		im_pred = cv2.cvtColor(np.array(im, dtype = np.uint8), cv2.COLOR_LAB2BGR)

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

def print_options(opt):
	message = ''
	message += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
	    comment = ''
	    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
	message += '----------------- End -------------------'
	print(message)

	# save to the disk
	expr_dir = os.path.join(opt.savedir)
	file_name = os.path.join(expr_dir, 'opt.txt')
	with open(file_name, 'wt') as opt_file:
		opt_file.write(message)
		opt_file.write('\n')
