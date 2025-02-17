# coding: utf-8

import os,argparse
import torch
import torch.nn as nn
from utils.training_template import Learner
from utils.logging import get_logger

class Tester(Learner):
	def __init__(self, logger, checkpoint_path):
		self.logger = logger
		self.device = torch.device('cpu')
		checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.
		save_dir = os.path.dirname(checkpoint_path)
		seed = checkpoint['random_seed']
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
		
		for attr_name,info in checkpoint['modules'].items():
			info.pop('state_dict')
			self.set_module(attr_name, **info)
		
		new_checkpoint = dict(modules=dict())
		for attr_name,info in checkpoint['modules'].items():
			module = getattr(self, attr_name)
			if isinstance(module, nn.DataParallel):
				module = module.module
			info['state_dict'] = module.state_dict()
			new_checkpoint['modules'][attr_name] = info
		torch.save(new_checkpoint, os.path.join(save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=0)))
		self.logger.info('Initial params successfully saved.')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file containing the trained model.')
	args = parser.parse_args()

	logger = get_logger()
	learner = Tester(logger, args.checkpoint_path)