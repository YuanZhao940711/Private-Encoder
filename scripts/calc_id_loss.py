from argparse import ArgumentParser
import time
import numpy as np
import os
import json
import sys
from PIL import Image
import multiprocessing as mp
import math
import torch
import torchvision.transforms as trans

sys.path.append(".")
sys.path.append("..")

from models.mtcnn.mtcnn import MTCNN
from models.encoders.model_irse import IR_101
from configs.paths_config import model_paths
CIRCULAR_FACE_PATH = model_paths['circular_face']


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def extract_on_paths(file_paths, record_path):
	facenet = IR_101(input_size=112)
	#facenet = IR_101(input_size=224)
	facenet.load_state_dict(torch.load(CIRCULAR_FACE_PATH))
	facenet.cuda()
	facenet.eval()
	mtcnn = MTCNN()
	id_transform = trans.Compose([
		trans.ToTensor(),
		trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	pid = mp.current_process().name
	print('\t{} is starting to extract on {} images'.format(pid, len(file_paths)))
	tot_count = len(file_paths)
	count = 0

	scores_dict = {}
	for idx, (res_path, _) in enumerate(file_paths):
		count += 1
		if count % 100 == 0:
			print('{} done with {}/{}'.format(pid, count, tot_count))
		input_im = Image.open(res_path)
		#input_im, _ = mtcnn.align(input_im)
		input_im = mtcnn.detect(input_im)
		"""
		if input_im is None:
			print('{} skipping {}'.format(pid, res_path))
			continue"""
		input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]
		
		scores = []
		for i, (_, gt_path) in enumerate(file_paths):
			if i == idx:
				continue
			result_im = Image.open(gt_path)
			#result_im, _ = mtcnn.align(result_im)
			result_im = mtcnn.detect(result_im)
			"""
			if result_im is None:
				print('{} skipping {}'.format(pid, gt_path))
				continue"""
			result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
			
			score = float(input_id.dot(result_id))
			#print("Individual score is: {}".format(score))
			scores.append(score)
		scores.sort()
		max_scores = scores[-2]
		#print("maximum different identity's similarity score is: {}".format(max_scores))
		with open(os.path.join(record_path, 'record.txt'), 'a') as f:
			f.write("No. {}-maximum different identity similarity score is: {}\n".format(idx,max_scores))
		scores_dict[os.path.basename(gt_path)] = max_scores
	return scores_dict


def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--num_threads', type=int, default=4)
	parser.add_argument('--data_path', type=str,  default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	#parser.add_argument('--record_path', type=str, default='./id_scores_record')
	args = parser.parse_args()
	return args

import glob
def run(args):
	file_paths = []
	
	image_paths = sorted(glob.glob(os.path.join(args.data_path, "*.jpg")) + glob.glob(os.path.join(args.data_path, "*.png")))
	gt_paths = sorted(glob.glob(os.path.join(args.gt_path, "*.jpg")) + glob.glob(os.path.join(args.gt_path, "*.png")))
	#print(image_paths,gt_paths)
	for img, gt in zip(image_paths,gt_paths):
		file_paths.append([img, gt])

	out_path = os.path.join(os.path.dirname(args.data_path), 'inference_metrics')
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	"""
	for f in os.listdir(args.data_path):
		image_path = os.path.join(args.data_path, f)
		gt_path = os.path.join(args.gt_path, f)
		if f.endswith(".jpg") or f.endswith('.png'):
			file_paths.append([image_path, gt_path.replace('.png','.jpg')])
	"""
	#print(file_paths)
	#file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
	#pool = mp.Pool(args.num_threads)
	print('Running on {} paths\nHere we goooo'.format(len(file_paths)))

	tic = time.time()
	#results = pool.map(extract_on_paths, file_chunks)
	scores_dict = extract_on_paths(file_paths, out_path)

	all_scores = list(scores_dict.values())
	mean = np.max(all_scores)
	std = np.std(all_scores)
	result_str = 'New Maximum score is {:.2f}+-{:.2f}'.format(mean, std)
	print(result_str)

	with open(os.path.join(out_path, 'stat_id.txt'), 'w') as f:
		f.write(result_str)
	with open(os.path.join(out_path, 'scores_id.json'), 'w') as f:
		json.dump(scores_dict, f)
	
	toc = time.time()
	print('Mischief managed in {}s'.format(toc - tic))


if __name__ == '__main__':
	args = parse_args()
	run(args)
