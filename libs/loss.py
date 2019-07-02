from numpy.random import randint
import torch
import torch.nn as nn

def weight_L1(pred, gt, w_a, w_b, mask=None):
	l1 = nn.L1Loss()
	if mask is not None:
		pred = pred * mask.repeat(1,pred.size(1),1,1)
		gt = gt * mask.repeat(1,gt.size(1),1,1)
	l_l = l1(pred[:,0,:,:], gt[:,0,:,:])
	l_a = l1(pred[:,1,:,:], gt[:,1,:,:])
	l_b = l1(pred[:,2,:,:], gt[:,2,:,:])
	loss = l_l + l_a * w_a + l_b * w_b
	return loss

def weightsingle_L1(pred, gt, w_a, w_b):
	n,c,h,w = pred.size()
	l_l = torch.abs(pred[:,0,:,:] - gt[:,0,:,:])
	l_a = torch.abs(pred[:,1,:,:] - gt[:,1,:,:])
	l_b = torch.abs(pred[:,2,:,:] - gt[:,2,:,:])
	
	l_l = torch.mean(l_l.view(n,-1),dim=1)
	l_a = torch.mean(l_a.view(n,-1),dim=1)
	l_b = torch.mean(l_b.view(n,-1),dim=1)
	loss = l_l + l_a * w_a + l_b * w_b
	return loss

def switch_L1_thr(pred2, frame2_var, pred1, frame1_var, w_a, w_b, thr):
	loss1 = weightsingle_L1(pred2, frame2_var, w_a, w_b)
	loss2 = weightsingle_L1(pred1, frame1_var, w_a, w_b)
	loss = loss1 + 0.1 * loss2
	loss[loss1 > thr] = 0
	return torch.mean(loss)

def L1_thr(pred2, frame2_var, w_a, w_b, thr):
	loss = weightsingle_L1(pred2, frame2_var, w_a, w_b)
	loss[loss > thr] = 0
	return torch.mean(loss)

# merge the above codes
def L1_loss(pred2, frame2_var, w_a, w_b, thr=None, pred1=None, frame1_var=None):
	if pred1 is None:
		loss = weightsingle_L1(pred2, frame2_var, w_a, w_b)
	else:
		loss1 = weightsingle_L1(pred2, frame2_var, w_a, w_b)
		loss2 = weightsingle_L1(pred1, frame1_var, w_a, w_b)
		loss = loss1 + 0.1 * loss2
	if thr is not None:	
		loss[loss > thr] = 0
	return torch.mean(loss)