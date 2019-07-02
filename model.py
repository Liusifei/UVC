import copy
import torch
import torch.nn as nn
from libs.net_utils import NLM, NLM_dot, NLM_woSoft
from torchvision.models import resnet18
from autoencoder import encoder3, decoder3, encoder_res18, encoder_res50
from torch.utils.serialization import load_lua
from libs.utils import *

def transform(aff, frame1):
	"""
	Given aff, copy from frame1 to construct frame2.
	INPUTS:
	 - aff: (h*w)*(h*w) affinity matrix
	 - frame1: n*c*h*w feature map
	"""
	b,c,h,w = frame1.size()
	frame1 = frame1.view(b,c,-1)
	frame2 = torch.bmm(frame1, aff)
	return frame2.view(b,c,h,w)

class normalize(nn.Module):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	"""

	def __init__(self, mean, std = (1.0,1.0,1.0)):
		super(normalize, self).__init__()
		self.mean = nn.Parameter(torch.FloatTensor(mean).cuda(), requires_grad=False)
		self.std = nn.Parameter(torch.FloatTensor(std).cuda(), requires_grad=False)

	def forward(self, frames):
		b,c,h,w = frames.size()
		frames = (frames - self.mean.view(1,3,1,1).repeat(b,1,h,w))/self.std.view(1,3,1,1).repeat(b,1,h,w)
		return frames

def create_flat_grid(F_size, GPU=True):
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
	grid[:,:,:,0] = (grid[:,:,:,0]+1)/2 * w
	grid[:,:,:,1] = (grid[:,:,:,1]+1)/2 * h
	grid_flat = grid.view(b,-1,2)
	if(GPU):
		grid_flat = grid_flat.cuda()
	return grid_flat


class track_match_comb(nn.Module):
	def __init__(self, pretrained, encoder_dir = None, decoder_dir = None, temp=1, Resnet = "r18", color_switch=True, coord_switch=True):
		super(track_match_comb, self).__init__()

		if Resnet in "r18":
			self.gray_encoder = encoder_res18(pretrained=pretrained, uselayer=4)
		elif Resnet in "r50":
			self.gray_encoder = encoder_res50(pretrained=pretrained, uselayer=4)
		self.rgb_encoder = encoder3(reduce=True)
		self.decoder = decoder3(reduce=True)

		self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		self.decoder.load_state_dict(torch.load(decoder_dir))
		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False

		self.nlm = NLM_woSoft()
		self.normalize = normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		self.softmax = nn.Softmax(dim=1)
		self.temp = temp
		self.grid_flat = None
		self.grid_flat_crop = None
		self.color_switch = color_switch
		self.coord_switch = coord_switch


	def forward(self, img_ref, img_tar, warm_up=True, patch_size=None):
		n, c, h_ref, w_ref = img_ref.size()
		n, c, h_tar, w_tar = img_tar.size()
		gray_ref = copy.deepcopy(img_ref[:,0].view(n,1,h_ref,w_ref).repeat(1,3,1,1))
		gray_tar = copy.deepcopy(img_tar[:,0].view(n,1,h_tar,w_tar).repeat(1,3,1,1))
		
		gray_ref = (gray_ref + 1) / 2
		gray_tar = (gray_tar + 1) / 2

		gray_ref = self.normalize(gray_ref)
		gray_tar = self.normalize(gray_tar)

		Fgray1 = self.gray_encoder(gray_ref)
		Fgray2 = self.gray_encoder(gray_tar)
		Fcolor1 = self.rgb_encoder(img_ref)

		output = []

		if warm_up:
			aff = self.nlm(Fgray1, Fgray2)
			aff_norm = self.softmax(aff)
			Fcolor2_est = transform(aff_norm, Fcolor1)
			color2_est = self.decoder(Fcolor2_est)
			
			output.append(color2_est)
			output.append(aff)
				
			if self.color_switch:
				Fcolor2 = self.rgb_encoder(img_tar)
				Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
				color1_est = self.decoder(Fcolor1_est)
				output.append(color1_est)
		else:
			if(self.grid_flat is None):
				self.grid_flat = create_flat_grid(Fgray2.size())
			aff_ref_tar = self.nlm(Fgray1, Fgray2)
			aff_ref_tar = torch.nn.functional.softmax(aff_ref_tar * self.temp, dim = 2)
			coords = torch.bmm(aff_ref_tar, self.grid_flat)
			new_c = coords2bbox(coords, patch_size, h_tar, w_tar)
			Fgray2_crop = diff_crop(Fgray2, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3], patch_size[1], patch_size[0])
			
			aff_p = self.nlm(Fgray1, Fgray2_crop)
			aff_norm = self.softmax(aff_p * self.temp)
			Fcolor2_est = transform(aff_norm, Fcolor1)
			color2_est = self.decoder(Fcolor2_est)
			
			Fcolor2_full = self.rgb_encoder(img_tar)
			Fcolor2_crop = diff_crop(Fcolor2_full, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3], patch_size[1], patch_size[0])

			output.append(color2_est)
			output.append(Fcolor2_crop)
			output.append(aff_p)
			output.append(new_c*8)
			output.append(coords)

			# color orthorganal
			if self.color_switch:
				Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2_crop)
				color1_est = self.decoder(Fcolor1_est)
				output.append(color1_est)

			# coord orthorganal
			if self.coord_switch:
				aff_norm_tran = self.softmax(aff_p.permute(0,2,1)*self.temp)
				if self.grid_flat_crop is None:
					self.grid_flat_crop = create_flat_grid(Fp_tar.size()).permute(0,2,1).detach()
				C12 = torch.bmm(self.grid_flat_crop, aff_norm)
				C11 = torch.bmm(C12, aff_norm_tran)
				output.append(self.grid_flat_crop)
				output.append(C11)

			# return 	pred1, pred2, aff_p, new_c * 8, self.grid_flat_crop, C11, coords
		return output


class Model_switchGTfixdot_swCC_Res(nn.Module):
	def __init__(self, encoder_dir = None, decoder_dir = None, fix_dec = True,
					   temp = None, pretrainRes = False, uselayer=3, model='resnet18'):
		'''
		For switchable concenration loss
		Using Resnet18
		'''
		super(Model_switchGTfixdot_swCC_Res, self).__init__()
		if(model == 'resnet18'):
			print('Use ResNet18.')
			self.gray_encoder = encoder_res18(pretrained = pretrainRes, uselayer=uselayer)
		else:
			print('Use ResNet50.')
			self.gray_encoder = encoder_res50(pretrained = pretrainRes, uselayer=uselayer)
		self.rgb_encoder = encoder3(reduce = True)
		self.nlm = NLM_woSoft()
		self.decoder = decoder3(reduce = True)
		self.temp = temp
		self.softmax = nn.Softmax(dim=1)
		self.cos_window = torch.Tensor(np.outer(np.hanning(40), np.hanning(40))).cuda()
		self.normalize = normalize(mean=[0.485, 0.456, 0.406],
								   std=[0.229, 0.224, 0.225])

		self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		self.decoder.load_state_dict(torch.load(decoder_dir))

		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False

	def forward(self, gray1, gray2, color1=None, color2=None):
		# move gray scale image to 0-1 so that they match ImageNet pre-training
		gray1 = (gray1 + 1) / 2
		gray2 = (gray2 + 1) / 2

		# normalize to fit resnet
		b = gray1.size(0)

		gray1 = self.normalize(gray1)
		gray2 = self.normalize(gray2)

		Fgray1 = self.gray_encoder(gray1)
		Fgray2 = self.gray_encoder(gray2)

		aff = self.nlm(Fgray1, Fgray2) # bx4096x4096
		aff_norm = self.softmax(aff*self.temp)

		if(color1 is None):
			return aff_norm, Fgray1, Fgray2

		Fcolor1 = self.rgb_encoder(color1)
		Fcolor2 = self.rgb_encoder(color2)
		Fcolor2_est = transform(aff_norm, Fcolor1)
		pred2 = self.decoder(Fcolor2_est)

		Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
		pred1 = self.decoder(Fcolor1_est)

		return pred1, pred2, aff_norm, aff, Fgray1, Fgray2	