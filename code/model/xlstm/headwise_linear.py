# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadWiseLinear(nn.Module):
	def __init__(self, in_features, out_features, nhead, bias=True):
		super().__init__()
		in_per_head = in_features//nhead
		out_per_head = out_features//nhead
		self.register_parameter('weight',
			nn.Parameter(torch.empty(nhead,out_per_head,in_per_head),requires_grad=True))
		nn.init.normal_(self.weight.data, std=math.sqrt(2/5/in_features))
		if bias:
			self.register_parameter('bias',
				nn.Parameter(torch.zeros(nhead,out_per_head)))
		else:
			self.bias = None
		self.in_features = in_features
			
	def forward(self, x):
		if x.size(-1)==self.in_features: # NOTE: Not headed yet
			x = x.view(*x.size()[:-1],self.weight.size(0),-1)
		x = torch.einsum('...hi,hoi->...ho', x, self.weight)
		if not self.bias is None:
			x = x + self.bias
		return x