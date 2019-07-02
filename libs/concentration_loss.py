import torch
import torch.nn as nn

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

def aff2coord(F_size, grid, A = None, temp = None, softmax=None):
	"""
	INPUT:
	 - A: a (H*W)*(H*W) affinity matrix
	 - F_size: image feature size
	 - mode: if mode is coord, return coordinates, else return flow
	 - grid: a standard grid, see create_grid function.
	OUTPUT:
	 - U: a (2*H*W) coordinate tensor, U_ij indicates the coordinates of pixel ij in target image.
	"""
	grid = grid.permute(0,3,1,2)
	if A is not None:
		if softmax is not None:
			if temp is None:
				raise Exception("Need temp for softmax!")
			A = softmax(A*temp)
		# b x c x h x w
		U = transform(A, grid)
	else:
		U = grid
	return U

def create_grid(F_size, GPU=True):
	b, c, h, w = F_size
	theta = torch.tensor([[1,0,0],[0,1,0]])
	theta = theta.unsqueeze(0).repeat(b,1,1)
	theta = theta.float()

	# grid is a uniform grid with left top (-1,1) and right bottom (1,1)
	# b * (h*w) * 2
	grid = nn.functional.affine_grid(theta, F_size)
	if(GPU):
		grid = grid.cuda()
	return grid

def im2col(img, win_len, stride=1):
	"""
	INPUTS:
	 - img: a b*c*h*w feature tensor.
	 - win_len: each pixel compares with its neighbors within a
				(win_len*2+1) * (win_len*2+1) window.
	OUTPUT:
	 - result: a b*c*(h*w)*(win_len*2+1)^2 tensor, unfolded neighbors for each pixel
	"""
	b,c,_,_ = img.size()
	# b * (c*w*w) * win_num
	unfold_img = torch.nn.functional.unfold(img, win_len, padding=0, stride=stride)
	unfold_img = unfold_img.view(b,c,win_len*win_len,-1)
	unfold_img = unfold_img.permute(0,1,3,2)
	return unfold_img

class ConcentrationLoss(nn.Module):
	def __init__(self, win_len, stride, F_size):
		super(ConcentrationLoss, self).__init__()
		self.win_len = win_len
		self.grid = nn.Parameter(create_grid(F_size), requires_grad = False)
		self.F_size = F_size
		self.stride = stride

	def forward(self, aff):
		b, c, h, w = self.F_size
		#if aff.dim() == 4:
		#	aff = torch.squeeze(aff)
		# b * 2 * h * w
		coord1 = aff2coord(self.F_size, self.grid, aff)
		# b * 2 * (h * w) * (win ^ 2)
		coord1_unfold = im2col(coord1, self.win_len, stride = self.stride)
		# b * 2 * (h * w) * 1
		# center = coord1_unfold[:,:,:,int((self.win_len ** 2)/2)]
		center = torch.mean(coord1_unfold, dim = 3)
		center = center.view(b, 2, -1, 1)
		# b * 2 * (h * w) * (win ^ 2)
		dis2center = (coord1_unfold - center) ** 2
		return torch.sum(dis2center) / dis2center.numel()

class ConcentrationDetachLoss(nn.Module):
	def __init__(self, win_len, stride, F_size):
		super(ConcentrationDetachLoss, self).__init__()
		self.win_len = win_len
		self.grid = nn.Parameter(create_grid(F_size), requires_grad = False)
		self.F_size = F_size
		self.stride = stride

	def forward(self, aff):
		b, c, h, w = self.F_size
		if aff.dim() == 4:
			aff = torch.squeeze(aff)
		# b * 2 * h * w
		coord1 = aff2coord(self.F_size, self.grid, aff)
		# b * 2 * (h * w) * (win ^ 2)
		coord1_unfold = im2col(coord1, self.win_len, stride = self.stride)
		# b * 2 * (h * w) * 1
		# center = coord1_unfold[:,:,:,int((self.win_len ** 2)/2)]
		center = torch.mean(coord1_unfold, dim = 3).detach()
		center = center.view(b, 2, -1, 1)
		# b * 2 * (h * w) * (win ^ 2)
		dis2center = (coord1_unfold - center) ** 2
		return torch.sum(dis2center) / dis2center.numel()

class ConcentrationSwitchLoss(nn.Module):
	def __init__(self, win_len, stride, F_size, temp):
		super(ConcentrationSwitchLoss, self).__init__()
		self.win_len = win_len
		self.grid = nn.Parameter(create_grid(F_size), requires_grad = False)
		self.F_size = F_size
		self.stride = stride
		self.temp = temp
		self.softmax = nn.Softmax(dim=1)

	def forward(self, aff):
		# aff here is not processed by softmax
		b, c, h, w = self.F_size
		if aff.dim() == 4:
			aff = torch.squeeze(aff)
		# b * 2 * h * w
		coord1 = aff2coord(self.F_size, self.grid, aff, self.temp, softmax=self.softmax)
		coord2 = aff2coord(self.F_size, self.grid, aff.permute(0,2,1), self.temp, softmax=self.softmax)
		# b * 2 * (h * w) * (win ^ 2)
		coord1_unfold = im2col(coord1, self.win_len, stride = self.stride)
		coord2_unfold = im2col(coord2, self.win_len, stride = self.stride)
		# b * 2 * (h * w) * 1
		center1 = torch.mean(coord1_unfold, dim = 3)
		center1 = center1.view(b, 2, -1, 1)
		# b * 2 * (h * w) * (win ^ 2)
		dis2center1 = (coord1_unfold - center1) ** 2
		# b * 2 * (h * w) * 1
		center2 = torch.mean(coord2_unfold, dim = 3)
		center2 = center2.view(b, 2, -1, 1)
		# b * 2 * (h * w) * (win ^ 2)
		dis2center2 = (coord2_unfold - center2) ** 2
		return (torch.sum(dis2center1) + torch.sum(dis2center2))/ dis2center1.numel()


if __name__ == '__main__':
	cl = ConcentrationLoss(win_len=8, F_size=torch.Size((1,3,32,32)), stride=8)
	aff = torch.Tensor(1,1024,1024)
	aff.uniform_()
	aff = aff.cuda()
	print(cl(aff))
