# coding: utf-8

import math
import torch
import torch.nn as nn

class Transformer(nn.Module):
	def __init__(self, d_model, *args, num_layers=1, encode_pos=True, is_causal=True, **kwargs):
		super().__init__()
		self.d_model = d_model
		self.is_causal = is_causal
		self.pos_enc = SinusoidalPositionEncoder(d_model) if encode_pos else None
		kwargs['batch_first'] = True
		layer = nn.TransformerEncoderLayer(d_model, *args, **kwargs)
		self.net = nn.TransformerEncoder(layer, num_layers=num_layers)

	def forward(self, x):
		"""
		reference: batch_size x seq_length x d_model
		"""
		if not self.pos_enc is None:
			x = x + self.pos_enc(x)
		mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), x.device) \
				if self.is_causal else None
		x = self.net(x, is_causal=self.is_causal, mask=mask)
		return x

class SinusoidalPositionEncoder(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		self.register_buffer('frequencies', frequencies)

	def forward(self, reference):
		"""
		reference: batch_size x seq_length x d_model
		"""
		position = torch.arange(reference.size(1), device=reference.device)
		angle = position[None,:,None] * self.frequencies[None,None,:]
		encoded = torch.cat([angle.sin(), angle.cos()], dim=-1)
		# encoded = encoded / math.sqrt(angle.size(-1)) # NOTE: L2-norm = 1.0
		return encoded