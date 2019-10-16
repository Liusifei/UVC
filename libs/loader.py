import numpy as np
import time
import random
import scipy.io
from PIL import Image
import cv2
import torch
from os.path import exists, join, split
import libs.transforms_multi as transforms
from torchvision import datasets

def video_loader(video_path, frame_end, step, frame_start=0):
	cap = cv2.VideoCapture(video_path)
	cap.set(1, frame_start - 1)
	video = []
	for i in range(frame_start - 1, frame_end, step):
		cap.set(1, i)
		success, image = cap.read()
		if not success:
			raise Exception('Error while reading video {}'.format(video_path))
		pil_im = image
		video.append(pil_im)
	return video


def framepair_loader(video_path, frame_start, frame_end):
	
	cap = cv2.VideoCapture(video_path)
	
	pair = []
	id_ = np.zeros(2)
	frame_num = frame_end - frame_start
	if frame_end > 50:
		id_[0] = random.randint(frame_start, frame_end-50)
		id_[1] = id_[0] + random.randint(1, 50)
	else:
		id_[0] = random.randint(frame_start, frame_end)
		id_[1] = random.randint(frame_start, frame_end)

	
	for ii in range(2):
		
		cap.set(1, id_[ii])
		
		success, image = cap.read()
		
		if not success:
			print("id, frame_end:", id_, frame_end)
			raise Exception('Error while reading video {}'.format(video_path))

		h,w,_ = image.shape
		h = (h // 64) * 64
		w = (w // 64) * 64
		image = cv2.resize(image, (w,h))
		image = image.astype(np.uint8)
		pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		pair.append(pil_im)
			
	return pair

def video_frame_counter(video_path):
	cap = cv2.VideoCapture(video_path)
	return cap.get(7)


class VidListv1(torch.utils.data.Dataset):
	# for warm up, random crop both
	def __init__(self, video_path, list_path, patch_size, rotate = 10, scale=1.2, is_train=True, moreaug= True):
		super(VidListv1, self).__init__()
		self.data_dir = video_path
		self.list_path = list_path
		normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))

		t = []
		if rotate > 0:
			t.append(transforms.RandomRotate(rotate))
		if scale > 0:
			t.append(transforms.RandomScale(scale))
		t.extend([transforms.RandomCrop(patch_size, seperate =moreaug), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
			  normalize])

		self.transforms = transforms.Compose(t)
		
		self.is_train = is_train
		self.read_list()

	def __getitem__(self, idx):
		while True:
			video_ = self.list[idx]
			frame_end = video_frame_counter(video_)-1
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(video_, 0, frame_end)
		data = list(self.transforms(*pair_))
		return tuple(data)

	def __len__(self):
		return len(self.list)

	def read_list(self):
		path = join(self.list_path)
		root = path.partition("Kinetices/")[0]
		if not exists(path):
			raise Exception("{} does not exist in kinet_dataset.py.".format(path))
		self.list = [line.replace("/Data/", root).strip() for line in open(path, 'r')]


class VidListv2(torch.utils.data.Dataset):
	# for localization, random crop frame1
	def __init__(self, video_path, list_path, patch_size, window_len, rotate = 10, scale = 1.2, full_size = 640, is_train=True):
		super(VidListv2, self).__init__()
		self.data_dir = video_path
		self.list_path = list_path
		self.window_len = window_len
		normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
		self.transforms1 = transforms.Compose([
						   transforms.RandomRotate(rotate),
						   # transforms.RandomScale(scale),
						   transforms.ResizeandPad(full_size),
						   transforms.RandomCrop(patch_size),
						   transforms.ToTensor(),
						   normalize])			
		self.transforms2 = transforms.Compose([
						   transforms.ResizeandPad(full_size),
						   transforms.ToTensor(),
						   normalize])
		self.is_train = is_train
		self.read_list()

	def __getitem__(self, idx):
		while True:
			video_ = self.list[idx]
			frame_end = video_frame_counter(video_)-1
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(video_, 0, frame_end)
		data1 = list(self.transforms1(*pair_))
		data2 = list(self.transforms2(*pair_))
		if self.window_len == 2:
			data = [data1[0],data2[1]]
		else:
			data = [data1[0],data2[1], data2[2]]
		return tuple(data)

	def __len__(self):
		return len(self.list)

	def read_list(self):
		path = join(self.list_path)
		root = path.partition("Kinetices/")[0]
		if not exists(path):
			raise Exception("{} does not exist in kinet_dataset.py.".format(path))
		self.list = [line.replace("/Data/", root).strip() for line in open(path, 'r')]



if __name__ == '__main__':
	normalize = transforms.Normalize(mean = (128, 128, 128),
									 std = (128, 128, 128))
	t = []
	t.extend([transforms.RandomCrop(256),
			  transforms.RandomHorizontalFlip(),
			  transforms.ToTensor(),
			  normalize])
	dataset_train = VidList('/home/xtli/DATA/compress/train_256/',
							'/home/xtli/DATA/compress/train.txt',
							transforms.Compose(t), window_len=2)

	train_loader = torch.utils.data.DataLoader(dataset_train,
											   batch_size = 16,
											   shuffle = True,
											   num_workers=8,
											   drop_last=True)

	start_time = time.time()
	for i, (frames) in enumerate(train_loader):
		print(i)
		if(i >= 1000):
			break
	end_time = time.time()
	print((end_time - start_time) / 1000)
