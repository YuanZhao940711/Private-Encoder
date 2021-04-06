"""
This file runs the main training/val loop
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
