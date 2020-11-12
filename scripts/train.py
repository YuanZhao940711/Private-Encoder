"""
This file runs the main training/val loop
"""

"""
training:
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=./experiment \
--stylegan_weights=model_paths['stylegan_ffhq']\
--checkpoint_path=None\
--workers=8 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=8 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1\
--max_steps=500000 \
--image_interval=100 \
--board_interval=50 \
--val_interval=2500 \
--save_interval=5000 
"""

import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

# import a class is not equal to call its __init__ function
from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()

	try:
		os.makedirs(opts.exp_dir)
		print("Creating folder... {}".format(opts.exp_dir))
	except:
		print("Folder {} already exists".format(opts.exp_dir))
		pass

	# vars()
	# Without arguments, equivalent to locals(). 
	# With an argument, equivalent to object.dict
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
