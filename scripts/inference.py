import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
	test_opts = TestOptions().parse()
	
	privacy_id = test_opts.checkpoint_path.split('/')[-2]
	folder_name = 'inference_results_'+privacy_id

	out_path_results = os.path.join(test_opts.exp_dir, folder_name)
	
	ori_save_dir = os.path.join(out_path_results, 'original')
	gen_save_dir = os.path.join(out_path_results, 'generated')
	
	os.makedirs(ori_save_dir, exist_ok=True)
	os.makedirs(gen_save_dir, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))
	transforms_dict = data_configs.DATASETS[opts.dataset_type](opts).get_transforms()
	
	dataset = InferenceDataset(root=opts.data_path,
	                           transform=transforms_dict['transform_inference'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=False)

	global_i = 0
	global_time = []
	for input_batch in tqdm(dataloader):
		with torch.no_grad():
			input_cuda = input_batch.cuda().float()
			tic = time.time()
			result_batch = run_on_batch(input_cuda, net, opts)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			ori_img = tensor2im(input_cuda[i])#.resize((128,128))
			gen_img = tensor2im(result_batch[i])#.resize((128,128))

			im_path = dataset.paths[global_i]
      
			ori_save_path = os.path.join(ori_save_dir, os.path.basename(im_path))
			gen_save_path = os.path.join(gen_save_dir, os.path.basename(im_path))

			Image.fromarray(np.array(ori_img)).save(ori_save_path)
			Image.fromarray(np.array(gen_img)).save(gen_save_path)

			global_i += 1

	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)


def run_on_batch(inputs, net, opts):
	if opts.latent_mask is None:
		result_batch = net(inputs, randomize_noise=False)
	else:
		latent_mask = [int(l) for l in opts.latent_mask.split(",")]
		result_batch = []
		for image_idx, input_image in enumerate(inputs):
			# get latent vector to inject into our input image
			vec_to_inject = np.random.randn(1, 512).astype('float32')
			_, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
			                          input_code=True,
			                          return_latents=True)
			# get output image with injected style vector
			res = net(input_image.unsqueeze(0).to("cuda").float(),
			          latent_mask=latent_mask,
			          inject_latent=latent_to_inject,
			          alpha=opts.mix_alpha)
			result_batch.append(res)
		result_batch = torch.cat(result_batch, dim=0)
	return result_batch


if __name__ == '__main__':
	run()
