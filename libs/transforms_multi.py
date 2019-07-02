import numbers
import random
import scipy.io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
	import accimage
except ImportError:
	accimage = None
import torch

class Scale(object):
	"""docstring for Scale"""
	def __init__(self, short_side):
		self.short_side = short_side

	def __call__(self, pair_0, pair_1):
		if(type(self.short_side) == int):
			h,w,c = pair_0.shape
			if(h > w):
				tw = self.short_side
				th = (tw * h) / w
				th = int((th // 64) * 64)
			else:
				th = self.short_side
				tw = (th * w) / h
				tw = int((tw // 64) * 64)
		elif(type(self.short_side) == list):
			th = self.short_side[0]
			tw = self.short_side[1]

		interpolation = cv2.INTER_NEAREST
		pair_0 = cv2.resize(pair_0, dsize = (tw, th), interpolation=interpolation)
		pair_1 = cv2.resize(pair_1, dsize = (tw, th), interpolation=interpolation)

		return pair_0, pair_1
		
class RandomCrop(object):
	"""for pair of frames"""
	def __init__(self, size, seperate = False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.seperate = seperate

	def __call__(self, *frames):
		frames = list(frames)
		h, w, c = frames[0].shape
		th, tw = self.size
		top = bottom = left = right = 0

		if w == tw and h == th:
			return frames

		if w < tw:
			left = (tw - w) // 2
			right = tw - w - left
		if h < th:
			top = (th - h) // 2
			bottom = th - h - top
		if left > 0 or right > 0 or top > 0 or bottom > 0:
			for i in range(len(frames)):
				frames[i] = pad_image(
				'reflection', frames[i], top, bottom, left, right)
		if w > tw:
			x1 = np.array([random.randint(0, w - tw)])
			x1 = np.concatenate((x1,x1))
			if self.seperate:
				#print("True")
				x1[1] = np.array([random.randint(0, w - tw)])
			for i in range(len(frames)):
				frames[i] = frames[i][:, x1[i]:x1[i]+tw]
		if h > th:
			y1 = np.array([random.randint(0, h - th)])
			y1 = np.concatenate((y1,y1))

			if self.seperate:
				y1[1] = np.array([random.randint(0, h - th)])

			for i in range(len(frames)):
				frames[i] = frames[i][y1[i]:y1[i]+th]

		return frames

class CenterCrop(object):
	"""for pair of frames"""
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size

	def __call__(self, *frames):
		frames = list(frames)
		h, w, c = frames[0].shape
		th, tw = self.size
		top = bottom = left = right = 0

		if w == tw and h == th:
			return frames

		if w < tw:
			left = (tw - w) // 2
			right = tw - w - left
		if h < th:
			top = (th - h) // 2
			bottom = th - h - top
		if left > 0 or right > 0 or top > 0 or bottom > 0:
			for i in range(len(frames)):
				frames[i] = pad_image(
				'reflection', frames[i], top, bottom, left, right)

		if w > tw:
			#x1 = random.randint(0, w - tw)
			x1 = (w - tw) // 2
			for i in range(len(frames)):
				frames[i] = frames[i][:, x1:x1+tw]
		if h > th:
			#y1 = random.randint(0, h - th)
			y1 = (h - th) // 2
			for i in range(len(frames)):
				frames[i] = frames[i][y1:y1+th]

		return frames

class RandomScale(object):
	"""docstring for RandomScale"""
	def __init__(self, scale, seperate = False):
		if isinstance(scale, numbers.Number):
			scale = [1 / scale, scale]
		self.scale = scale
		self.seperate = seperate

	def __call__(self, *frames):
		h,w,c = frames[0].shape
		results = []
		if self.seperate:
			ratio1 = random.uniform(self.scale[0], self.scale[1])
			ratio2 = random.uniform(self.scale[0], self.scale[1])
			tw1 = int(ratio1*w)
			th1 = int(ratio1*h)
			tw2 = int(ratio2*w)
			th2 = int(ratio2*h)

			if ratio1 == 1:
				results.append(frames[0])
			elif ratio1 < 1:
				interpolation = cv2.INTER_LANCZOS4
			elif ratio1 > 1:
				interpolation = cv2.INTER_CUBIC	
			frame = cv2.resize(frames[0], dsize = (tw1, th1), interpolation=interpolation)
			results.append(frame)

			if ratio2 == 1:
				results.append(frames[1])
			elif ratio2 < 1:
				interpolation = cv2.INTER_LANCZOS4
			elif ratio2 > 1:
				interpolation = cv2.INTER_CUBIC	
			frame = cv2.resize(frames[1], dsize = (tw2, th2), interpolation=interpolation)
			results.append(frame)
		else:
			ratio = random.uniform(self.scale[0], self.scale[1])
			tw = int(ratio*w)
			th = int(ratio*h)
			if ratio == 1:
				return frames
			elif ratio < 1:
				interpolation = cv2.INTER_LANCZOS4
			elif ratio > 1:
				interpolation = cv2.INTER_CUBIC
			for frame in frames:
				frame = cv2.resize(frame, dsize = (tw, th), interpolation=interpolation)
				results.append(frame)
		# print(results[0].shape,type(results[1]))
		return results

class RandomRotate(object):
	"""docstring for RandomRotate"""
	def __init__(self, angle, seperate = False):
		self.angle = angle
		self.seperate = seperate

	#def __call__(self, pair_0, pair_1):
	def __call__(self, *frames):
		results = []
		if self.seperate:
			angle = random.randint(0, self.angle * 2) - self.angle
			h,w,c = frames[0].shape
			p = max((h, w))
			frame = pad_image('reflection', frames[0], h,h,w,w)
			frame = rotatenumpy(frame, angle)
			frame = frame[h : h + h, w : w + w]
			results.append(frame)

			angle = random.randint(0, self.angle * 2) - self.angle
			h,w,c = frames[1].shape
			p = max((h, w))
			frame = pad_image('reflection', frames[1], h,h,w,w)
			frame = rotatenumpy(frame, angle)
			frame = frame[h : h + h, w : w + w]
			results.append(frame)
		else:
			angle = random.randint(0, self.angle * 2) - self.angle
			for frame in frames:
				h,w,c = frame.shape
				p = max((h, w))
				frame = pad_image('reflection', frame, h,h,w,w)
				frame = rotatenumpy(frame, angle)
				frame = frame[h : h + h, w : w + w]
				results.append(frame)

		return results

class RandomHorizontalFlip(object):
	"""Randomly horizontally flips the given PIL.Image with a probability of 0.5
	"""

	def __call__(self, *frames):
		results = []
		if random.random() < 0.5:
			for frame in frames:
				results.append(cv2.flip(frame, 1))
		else:
			results = frames
		return results

class Resize(object):
	"""Resize the input PIL Image to the given size.
	Args:
		size (sequence or int): Desired output size. If size is a sequence like
			(h, w), output size will be matched to this. If size is an int,
			smaller edge of the image will be matched to this number.
			i.e, if height > width, then image will be rescaled to
			(size * height / width, size)
		interpolation (int, optional): Desired interpolation. Default is
			``PIL.Image.BILINEAR``
	"""

	def __init__(self, size, interpolation=cv2.INTER_NEAREST):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, pair_0, pair_1):

		return resize(pair_0, self.size, self.interpolation), \
				resize(pair_1, self.size, self.interpolation)


class Pad(object):

	def __init__(self, padding, fill=0):
		assert isinstance(padding, numbers.Number)
		assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
			   isinstance(fill, tuple)
		self.padding = padding
		self.fill = fill

	def __call__(self, pair_0, pair_1):
		if self.fill == -1:
			pair_0 = pad_image('reflection', pair_0,
				self.padding, self.padding, self.padding, self.padding)
			pair_1 = pad_image('reflection', pair_1,
				self.padding, self.padding, self.padding, self.padding)
		else:
			pair_0 = pad_image('constant', pair_0,
				self.padding, self.padding, self.padding, self.padding,
				value=self.fill)
			pair_1 = pad_image('constant', pair_1,
				self.padding, self.padding, self.padding, self.padding,
				value=self.fill)

		return pair_0, pair_1


class ResizeandPad(object):
	"""
	resize the larger boundary to the desized eva_size;
	pad the smaller one to square
	"""
	def __init__(self, size, interpolation=cv2.INTER_NEAREST):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, pair_0, pair_1):
		"""
		Resize and Pad
		"""
		pair_0 = resize_large(pair_0, self.size, self.interpolation)
		pair_1 = resize_large(pair_1, self.size, self.interpolation)
		h,w,_ = pair_0.shape
		if w > h:
			bd = int((w - h) / 2)
			pair_0 = pad_image('reflection', pair_0, bd, (w-h)-bd, 0, 0)
			# pair_1 = pad_image('reflection', pair_1, bd, (w-h)-bd, 0, 0)
		elif h > w:
			bd = int((h-w) / 2)
			pair_0 = pad_image('reflection', pair_0, 0, 0, bd, (h-w)-bd)
			# pair_1 = pad_image('reflection', pair_1, 0, 0, bd, (h-w)-bd)
		
		h,w,_ = pair_1.shape
		if w > h:
			bd = int((w - h) / 2)
			# pair_0 = pad_image('reflection', pair_0, bd, (w-h)-bd, 0, 0)
			pair_1 = pad_image('reflection', pair_1, bd, (w-h)-bd, 0, 0)
		elif h > w:
			bd = int((h-w) / 2)
			# pair_0 = pad_image('reflection', pair_0, 0, 0, bd, (h-w)-bd)
			pair_1 = pad_image('reflection', pair_1, 0, 0, bd, (h-w)-bd)
			
		return pair_0, pair_1


class Normalize(object):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	"""

	def __init__(self, mean, std = (1.0,1.0,1.0)):
		self.mean = torch.FloatTensor(mean)
		self.std = torch.FloatTensor(std)

	def __call__(self, *frames):
		results = []
		for frame in frames:
			for t, m, s in zip(frame, self.mean, self.std):
				t.sub_(m).div_(s)
			results.append(frame)
		return results


class ToTensor(object):
	"""Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	"""
	def __call__(self, *frames):
		results = []
		for frame in frames:
			frame = torch.from_numpy(frame.transpose(2,0,1).copy()).contiguous().float()
			results.append(frame)
		return results


class Compose(object):
	"""
	Composes several transforms together.
	"""
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *args):
		for t in self.transforms:
			args = t(*args)
		return args


#=============================functions===============================

def resize(img, size, interpolation=cv2.INTER_NEAREST):

	if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
		raise TypeError('Got inappropriate size arg: {}'.format(size))

	h, w = img.shape

	if isinstance(size, int):
		if (w <= h and w == size) or (h <= w and h == size):
			return img
		if w < h:
			ow = size
			oh = int(size * h / w)
			return cv2.resize(img, (ow, oh), interpolation)
		else:
			oh = size
			ow = int(size * w / h)
			return cv2.resize(img, (ow, oh), interpolation)
	else:
		return cv2.resize(img, size[::-1], interpolation)
		

def resize_large(img, size, interpolation=cv2.INTER_NEAREST):

	if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
		raise TypeError('Got inappropriate size arg: {}'.format(size))

	h, w,_ = img.shape

	if isinstance(size, int):
		if (w >= h and w == size) or (h >= w and h == size):
			return img
		if w > h:
			ow = size
			oh = int(size * h / w)
			return cv2.resize(img, (ow, oh), interpolation)
		else:
			oh = size
			ow = int(size * w / h)
			return cv2.resize(img, (ow, oh), interpolation)
	else:
		return cv2.resize(img, size[::-1], interpolation)


def rotatenumpy(image, angle, interpolation=cv2.INTER_NEAREST):
	rot_mat = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]), flags=interpolation)
	return result

# good, written with numpy
def pad_reflection(image, top, bottom, left, right):
	if top == 0 and bottom == 0 and left == 0 and right == 0:
		return image
	h, w = image.shape[:2]
	next_top = next_bottom = next_left = next_right = 0
	if top > h - 1:
		next_top = top - h + 1
		top = h - 1
	if bottom > h - 1:
		next_bottom = bottom - h + 1
		bottom = h - 1
	if left > w - 1:
		next_left = left - w + 1
		left = w - 1
	if right > w - 1:
		next_right = right - w + 1
		right = w - 1
	new_shape = list(image.shape)
	new_shape[0] += top + bottom
	new_shape[1] += left + right
	new_image = np.empty(new_shape, dtype=image.dtype)
	new_image[top:top+h, left:left+w] = image
	new_image[:top, left:left+w] = image[top:0:-1, :]
	new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
	new_image[:, :left] = new_image[:, left*2:left:-1]
	new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
	return pad_reflection(new_image, next_top, next_bottom,
						  next_left, next_right)

# good, writen with numpy
def pad_constant(image, top, bottom, left, right, value):
	if top == 0 and bottom == 0 and left == 0 and right == 0:
		return image
	h, w = image.shape[:2]
	new_shape = list(image.shape)
	new_shape[0] += top + bottom
	new_shape[1] += left + right
	new_image = np.empty(new_shape, dtype=image.dtype)
	new_image.fill(value)
	new_image[top:top+h, left:left+w] = image
	return new_image

# change to np/non-np options
def pad_image(mode, image, top, bottom, left, right, value=0):
	if mode == 'reflection':
		return pad_reflection(image, top, bottom, left, right)
	elif mode == 'constant':
		return pad_constant(image, top, bottom, left, right, value)
	else:
		raise ValueError('Unknown mode {}'.format(mode))
