import torch
import torch.nn as nn

class LayerNorm(nn.Module):

	def __init__(self, eps=1e-5):
		super().__init__()
		self.register_parameter('gamma', None)
		self.register_parameter('beta', None)
		self.eps = eps

	def forward(self, x):
		if self.gamma is None:
			self.gamma = nn.Parameter(torch.ones(x.size()).cuda())
		if self.beta is None:
			self.beta = nn.Parameter(torch.zeros(x.size()).cuda())
		mean = torch.min(x, 1, keepdim=True)[0]
		std = x.std(1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class NLM(nn.Module):
	"""NLM layer, output affinity"""
	def __init__(self, is_norm = False, iso1 = True):
		super(NLM, self).__init__()
		self.is_norm = is_norm
		if is_norm:
			self.norm = LayerNorm()
		self.softmax = nn.Softmax(dim=1)
		self.iso1 = iso1

	def forward(self, in1, in2, return_unorm=False):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,N)
		in2 = in2.view(n,c,N)
		affinity = torch.bmm(in1.permute(0,2,1), in2)

		for ii in range(n):
			if self.iso1:
				affinity[ii] = affinity[ii] - 0.5*torch.diag(affinity[ii]).view(-1,1).repeat(1,N) - 0.5*torch.diag(affinity[ii]).view(1,-1).repeat(N,1)
			else:
				diag_ = torch.diag(affinity[ii])
				for xx in range(N):
					affinity[ii,xx] -= 0.5 * diag_
				for yy in range(N):
					affinity[ii, :, yy] -= 0.5 * diag_
		aff = self.softmax(affinity)
		if(return_unorm):
			return aff,affinity
		else:
			return aff

def featureL2Norm(feature):
	epsilon = 1e-6
	norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
	return torch.div(feature,norm)

class NLM_dot(nn.Module):
	"""NLM layer, output affinity"""
	def __init__(self, is_norm = False, temp = 1, l2_norm = False):
		super(NLM_dot, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		if is_norm:
			self.norm = LayerNorm()
		self.softmax = nn.Softmax(dim=1)
		self.temp = temp
		print("=============================================")
		print("==> Using dot non-local mean.          ")
		print("==> Temperature of softmax is {:2f}.".format(temp))
		if(l2_norm):
			print("==> Using L2 normalization.")
		print("=============================================")

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,N)
		in2 = in2.view(n,c,N)
		if self.is_norm:
			in1 = self.norm(in1)
			in2 = self.norm(in2)

		if self.l2_norm:
			in1 = featureL2Norm(in1)
			in2 = featureL2Norm(in2)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		affinity = self.softmax(affinity*self.temp) # n*N*N
		return affinity

class NLM_woSoft(nn.Module):
	"""NLM layer, output affinity, no softmax"""
	def __init__(self, is_norm = False, l2_norm = False):
		super(NLM_woSoft, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		if is_norm:
			self.norm = LayerNorm()
		# self.temp = temp
		print("=============================================")
		print("==> Using dot non-local mean.          ")
		# print("==> Temperature of softmax is {:2f}.".format(temp))
		if(l2_norm):
			print("==> Using L2 normalization.")
		print("=============================================")

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,-1)
		in2 = in2.view(n,c,-1)
		if self.is_norm:
			in1 = self.norm(in1)
			in2 = self.norm(in2)

		if self.l2_norm:
			in1 = featureL2Norm(in1)
			in2 = featureL2Norm(in2)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		# affinity = self.softmax(affinity*self.temp) # n*N*N
		return affinity