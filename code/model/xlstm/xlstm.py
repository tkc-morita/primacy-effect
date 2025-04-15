# coding: utf-8

import torch
import torch.nn as nn
from .slstm import sLSTMBlock
from .mlstm import mLSTMBlock

class xLSTM(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, num_layers=1, dropout=0.0):
		super().__init__()
		layers = list()
		for _ in range(num_layers):
			layers.append(mLSTMBlock(d_model, nhead, dropout, num_layers*2))
			layers.append(sLSTMBlock(d_model, nhead, dim_feedforward, dropout))
		self.layers = nn.Sequential(*layers)

	def forward(self, input):
		output = self.layers(input)
		return output