# coding: utf-8

import torch
import torch.nn as nn
from .s4 import S4Block,SSMKernelDPLR


class S4(nn.Module):
	def __init__(self, *args, num_layers=1, input_length=None, **kwargs):
		super().__init__()
		self.layers = nn.ModuleList([S4Block(*args, transposed=False, **kwargs)
										for l in range(num_layers)])
		if (not input_length is None) \
			and isinstance(self.layers[0].layer.kernel, SSMKernelDPLR) \
			and self.layers[0].layer.kernel.l_kernel.item()==0:
			for layer in self.layers:
				layer.layer.kernel._setup_C(input_length) # NOTE: Calling _setup_C inside nn.DataParallel raises an error.

	def forward(self, x):
		# states = []
		for layer in self.layers:
			x,state = layer(x) # NOTE: Strip off the layer states
			# states.append(state)
		return x#,torch.stack(states, dim=0)