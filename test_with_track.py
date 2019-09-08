# OS libraries
import os
import cv2
import glob
import copy
import math
import queue
import argparse
import scipy.misc
import numpy as np
from tqdm import tqdm
from PIL import Image

# Pytorch libraries
import torch
import torch.nn as nn

# Customized libraries
from libs.test_utils import *
from libs.model import transform
from libs.vis_utils import norm_mask
import libs.transforms_pair as transforms
from libs.model import Model_switchGTfixdot_swCC_Res as Model
from libs.track_utils import seg2bbox, draw_bbox, match_ref_tar
from libs.track_utils import squeeze_all, seg2bbox_v2, bbox_in_tar_scale

############################## helper functions ##############################

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type = int, default = 1,
	                    help = "batch size")
	parser.add_argument("-o","--out_dir", type = str,default="results_with_track/",
						help = "output path")
	parser.add_argument("--device", type = int, default = 5,
						help="0~4 for single GPU, 5 for dataparallel.")
	parser.add_argument("-c","--checkpoint_dir",type = str,
						default = "weights/checkpoint_latest.pth.tar",
						help = "checkpoints path")
	parser.add_argument("-s", "--scale_size", type = int, nargs = '+',
						help = "scale size, either a single number for short edge, or a pair for height and width")
	parser.add_argument("--pre_num", type = int, default = 7,
						help = "preceding frame numbers")
	parser.add_argument("--temp", type = float, default = 1,
						help = "softmax temperature")
	parser.add_argument("-t", "--topk", type = int, default = 5,
						help = "accumulate label from top k neighbors")
	parser.add_argument("-d", "--davis_dir", type = str,
						default = "/workspace/DAVIS/",
						help = "davis dataset path")

	print("Begin parser arguments.")
	args = parser.parse_args()
	args.is_train = False

	args.multiGPU = args.device == 5
	if not args.multiGPU:
		torch.cuda.set_device(args.device)

	args.val_txt = os.path.join(args.davis_dir, "ImageSets/2017/val.txt")
	args.davis_dir = os.path.join(args.davis_dir, "JPEGImages/480p/")
	return args

def vis_bbox(im, bbox, name, coords, seg):
	im = im * 128 + 128
	im = im.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
	im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
	fg_idx = seg.nonzero()
	im = draw_bbox(im, bbox, (0,0,255))

	for cnt in range(coords.size(0)):
		coord_i = coords[cnt]

		cv2.circle(im, (int(coord_i[0]*8), int(coord_i[1]*8)), 2, (0,255,0), thickness=-1)
	cv2.imwrite(name, im)

############################## tracking functions ##############################

def adjust_bbox(bbox_now, bbox_pre, a, h, w):
	"""
	Adjust a bounding box w.r.t previous frame,
	assuming objects don't go under abrupt changes.
	"""
	for cnt in bbox_pre.keys():
		if(cnt == 0):
			continue
		if(cnt in bbox_now and bbox_pre[cnt] is not None and bbox_now[cnt] is not None):
			bbox_now_h = (bbox_now[cnt].top  + bbox_now[cnt].bottom) / 2.0
			bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0

			bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
			bbox_now_width_  = bbox_now[cnt].right  - bbox_now[cnt].left

			bbox_pre_height = bbox_pre[cnt].bottom - bbox_pre[cnt].top
			bbox_pre_width  = bbox_pre[cnt].right  - bbox_pre[cnt].left

			bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
			bbox_now_width  = a * bbox_now_width_  + (1 - a) * bbox_pre_width

			bbox_now[cnt].left   = math.floor(bbox_now_w - bbox_now_width / 2.0)
			bbox_now[cnt].right  = math.ceil(bbox_now_w + bbox_now_width / 2.0)
			bbox_now[cnt].top    = math.floor(bbox_now_h - bbox_now_height / 2.0)
			bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)

			bbox_now[cnt].left = max(0, bbox_now[cnt].left)
			bbox_now[cnt].right = min(w, bbox_now[cnt].right)
			bbox_now[cnt].top = max(0, bbox_now[cnt].top)
			bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)

	return bbox_now

def bbox_next_frame(img_ref, seg_ref, img_tar, bbox_ref):
	"""
	Match bbox from the reference frame to the target frame
	"""
	F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
	seg_ref = seg_ref.squeeze(0)
	F_ref, F_tar = squeeze_all(F_ref, F_tar)
	c, h, w = F_ref.size()

	# get coordinates of each point in the target frame
	coords_ref_tar = match_ref_tar(F_ref, F_tar, seg_ref, args.temp)
	# coordinates -> bbox
	bbox_tar = bbox_in_tar_scale(coords_ref_tar, bbox_ref, h, w)
	# adjust bbox
	bbox_tar = adjust_bbox(bbox_tar, bbox_ref, 0.1, h, w)
	return bbox_tar, coords_ref_tar

def recoginition(img_ref, img_tar, bbox_ref, bbox_tar, seg_ref, model):
	"""
	propagate from bbox in the reference frame to bbox in the target frame
	"""
	F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
	seg_ref = seg_ref.squeeze()
	_, c, h, w = F_tar.size()
	seg_pred = torch.zeros(seg_ref.size())

	# calculate affinity only once to save time
	aff_whole = torch.mm(F_ref.view(c,-1).permute(1,0), F_tar.view(c,-1))
	aff_whole = torch.nn.functional.softmax(aff_whole * args.temp, dim=0)

	for cnt, br in bbox_ref.items():
		if not (cnt in bbox_tar):
			continue
		bt = bbox_tar[cnt]
		if(br is None or bt is None):
			continue
		seg_cnt = seg_ref[cnt]

		# affinity between two patches
		seg_ref_box = seg_cnt[br.top:br.bottom, br.left:br.right]
		seg_ref_box = seg_ref_box.unsqueeze(0).unsqueeze(0)

		h, w = F_ref.size(2), F_ref.size(3)
		mask = torch.zeros(h,w)
		mask[br.top:br.bottom, br.left:br.right] = 1
		mask = mask.view(-1)
		aff_row = aff_whole[mask.nonzero().squeeze(), :]

		h, w = F_tar.size(2), F_tar.size(3)
		mask = torch.zeros(h,w)
		mask[bt.top:bt.bottom, bt.left:bt.right] = 1
		mask = mask.view(-1)
		aff = aff_row[:, mask.nonzero().squeeze()]

		aff = aff.unsqueeze(0)

		seg_tar_box = transform_topk(aff,seg_ref_box.cuda(),k=args.topk,
					  h2 = bt.bottom - bt.top,w2 = bt.right - bt.left)
		seg_pred[cnt, bt.top:bt.bottom, bt.left:bt.right] = seg_tar_box

	return seg_pred

def disappear(seg,bbox_ref,bbox_tar=None):
	"""
	Check if bbox disappear in the target frame.
	"""
	b,c,h,w = seg.size()
	for cnt in range(c):
		if(torch.sum(seg[:,cnt,:,:]) < 3 or (not (cnt in bbox_ref))):
			return True
		if(bbox_ref[cnt] is None):
			return True
		if(bbox_ref[cnt].right - bbox_ref[cnt].left < 3 or bbox_ref[cnt].bottom - bbox_ref[cnt].top < 3):
			return True

		if(bbox_tar is not None):
			if(cnt not in bbox_tar.keys()):
				return True
			if(bbox_tar[cnt] is None):
				return True
			if(bbox_tar[cnt].right - bbox_tar[cnt].left < 3 or bbox_tar[cnt].bottom - bbox_tar[cnt].top < 3):
				return True
	return False

############################## testing functions ##############################

def forward(frame1, frame2, model, seg, return_feature=False):
	n, c, h, w = frame1.size()
	frame1_gray = frame1[:,0].view(n,1,h,w)
	frame2_gray = frame2[:,0].view(n,1,h,w)
	frame1_gray = frame1_gray.repeat(1,3,1,1)
	frame2_gray = frame2_gray.repeat(1,3,1,1)

	output = model(frame1_gray, frame2_gray, frame1, frame2)
	if(return_feature):
		return output[-2], output[-1]

	aff = output[2]
	frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)

	return frame2_seg

def test(model, frame_list, video_dir, first_seg, large_seg, first_bbox, seg_ori):
	video_dir = os.path.join(video_dir)
	video_nm = video_dir.split('/')[-1]
	video_folder = os.path.join(args.out_dir, video_nm)
	os.makedirs(video_folder, exist_ok = True)
	os.makedirs(os.path.join(video_folder, 'track'), exist_ok = True)

	transforms = create_transforms()

	# The queue stores `pre_num` preceding frames
	que = queue.Queue(args.pre_num)

	# frame 1
	frame1, ori_h, ori_w = read_frame(frame_list[0], transforms, args.scale_size)
	n, c, h, w = frame1.size()

	# saving first segmentation
	out_path = os.path.join(video_folder,"00000.png")
	imwrite_indexed(out_path, seg_ori)

	coords = first_seg[0,1].nonzero()
	coords = coords.flip(1)

	for cnt in tqdm(range(1,len(frame_list))):
		frame_tar, ori_h, ori_w = read_frame(frame_list[cnt], transforms, args.scale_size)

		with torch.no_grad():

			tmp_list = list(que.queue)
			if(len(tmp_list) > 0):
				pair = tmp_list[-1]
				framei = pair[0]
				segi = pair[1]
				bbox_pre = pair[2]
			else:
				bbox_pre = first_bbox
				framei = frame1
				segi = first_seg
			_, segi_int = torch.max(segi, dim=1)
			segi = to_one_hot(segi_int)
			bbox_tar, coords_ref_tar = bbox_next_frame(framei, segi, frame_tar, bbox_pre)

			if(bbox_tar is not None):

				if(1 in bbox_tar):
					tmp = copy.deepcopy(bbox_tar[1])
					if(tmp is not None):
						tmp.upscale(8)
						vis_bbox(frame_tar, tmp, os.path.join(video_folder, 'track', 'frame'+str(cnt+1)+'.png'), coords_ref_tar[1], segi[0,1,:,:])
				frame_tar_acc = recoginition(frame1, frame_tar, first_bbox, bbox_tar, first_seg, model)
			else:
				frame_tar_acc = forward(frame1, frame_tar, model, first_seg)
			frame_tar_acc = frame_tar_acc.cpu()


			# previous 7 frames
			tmp_queue = list(que.queue)
			for pair in tmp_queue:
				framei = pair[0]
				segi = pair[1]
				bboxi = pair[2]

				if(bbox_tar is None or disappear(segi, bboxi, bbox_tar)):
					frame_tar_est_i = forward(framei, frame_tar, model, segi)
					frame_tar_est_i = frame_tar_est_i.cpu()
				else:

					frame_tar_est_i = recoginition(framei, frame_tar, bboxi, bbox_tar, segi, model)

				frame_tar_acc += frame_tar_est_i.cpu().view(frame_tar_acc.size())
			frame_tar_avg = frame_tar_acc / (1 + len(tmp_queue))

		frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg",".png")
		out_path = os.path.join(video_folder,frame_nm)

		# upsampling & argmax
		if(frame_tar_avg.dim() == 3):
			frame_tar_avg = frame_tar_avg.unsqueeze(0)
		elif(frame_tar_avg.dim() == 2):
			frame_tar_avg = frame_tar_avg.unsqueeze(0).unsqueeze(0)
		frame_tar_up = torch.nn.functional.interpolate(frame_tar_avg,scale_factor=8,mode='bilinear')

		frame_tar_up = frame_tar_up.squeeze()
		frame_tar_up = norm_mask(frame_tar_up.squeeze())
		_, frame_tar_seg = torch.max(frame_tar_up.squeeze(), dim=0)

		frame_tar_seg = frame_tar_seg.squeeze().cpu().numpy()
		frame_tar_seg = np.array(frame_tar_seg, dtype=np.uint8)
		frame_tar_seg = scipy.misc.imresize(frame_tar_seg, (ori_h, ori_w), "nearest")
		imwrite_indexed(out_path,frame_tar_seg)

		if(que.qsize() == args.pre_num):
			que.get()
		seg = copy.deepcopy(frame_tar_avg.squeeze())
		frame, ori_h, ori_w = read_frame(frame_list[cnt], transforms, args.scale_size)
		bbox_tar = seg2bbox_v2(frame_tar_up.cpu(), bbox_pre)
		bbox_tar = adjust_bbox(bbox_tar, bbox_pre, 0.1, h, w)
		que.put([frame,seg.unsqueeze(0),bbox_tar])

if(__name__ == '__main__'):
	args = parse_args()
	with open(args.val_txt) as f:
		lines = f.readlines()
	f.close()

	model = Model(pretrainRes=False, temp = args.temp, uselayer=4)
	if(args.multiGPU):
		model = nn.DataParallel(model)
	checkpoint = torch.load(args.checkpoint_dir)
	best_loss = checkpoint['best_loss']
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{} ({})' (epoch {})"
		  .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
	model.cuda()
	model.eval()

	for cnt,line in enumerate(lines):
		video_nm = line.strip()
		print('[{:n}/{:n}] Begin to segmentate video {}.'.format(cnt,len(lines),video_nm))

		video_dir = os.path.join(args.davis_dir, video_nm)
		frame_list = read_frame_list(video_dir)
		seg_dir = frame_list[0].replace("JPEGImages","Annotations")
		seg_dir = seg_dir.replace("jpg","png")
		large_seg, first_seg, seg_ori = read_seg(seg_dir, args.scale_size)

		first_bbox = seg2bbox(large_seg, margin=0.6)
		for k,v in first_bbox.items():
			v.upscale(0.125)

		test(model, frame_list, video_dir, first_seg, large_seg, first_bbox, seg_ori)
