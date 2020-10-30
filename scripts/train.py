"""
This file runs the main training/val loop
"""

"""
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1
"""

import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
