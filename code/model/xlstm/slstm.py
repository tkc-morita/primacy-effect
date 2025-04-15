# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .headwise_linear import HeadWiseLinear

class sLSTMBlock(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, dropout):
		super().__init__()
		self.ln1 = nn.LayerNorm(d_model)
		self.slstm = sLSTMCore(d_model, nhead, dropout)
		self.ln2 = nn.LayerNorm(d_model)
		self.ff = FeedForward(d_model, dim_feedforward)
		self.d_model = d_model
		self.nhead = nhead

	def forward(self, input, hidden=None):
		output = self.slstm(self.ln1(input), hidden)[0] + input
		output = self.ff(self.ln2(output)) + output
		return output

class sLSTMCore(nn.Module):
	def __init__(self, d_model, nhead, dropout):
		# NOTE: Just like an LSTM layer but only a single bias.
		super().__init__()
		dim_per_head = d_model//nhead
		self.d_model = d_model
		self.nhead = nhead
		self.linear_ih = HeadWiseLinear(d_model,4*d_model,nhead)
		self._init_forget_bias()
		self.linear_hh = nn.Linear(dim_per_head,4*dim_per_head,bias=False)
		self.dropout = nn.Dropout(dropout)
		self.headwise_norm = nn.LayerNorm(dim_per_head)


	def _init_forget_bias(self):
		# NOTE: forget gate needs special initialization.
		init_vals = torch.linspace(3.0, 6.0, self.d_model
						).view(self.nhead,1,-1)
		init_vals = F.pad(init_vals, (0,0,1,2)) # Zeros for input gate and (cell input,output gate).
		with torch.no_grad():
			self.linear_ih.bias.copy_(init_vals.view(self.nhead,-1))

	def forward(self, input, hidden=None):
		"""
		input: batch_size x seq_length x d_model
		hidden: batch_size {x,*} nhead x d_model//nhead
		"""
		B,L,D = input.size()
		if hidden is None:
			h = torch.zeros(B, self.nhead, self.d_model//self.nhead, device=input.device)
			c = torch.zeros_like(h)
		else:
			h,c = hidden
		input = self.linear_ih(input) # -> B x L x H x 4*D/H
		m = 0.0
		n = 0.0
		output = list()
		for proj_t in input.unbind(dim=1):
			# Add the recurrent gains
			proj_t = proj_t + self.linear_hh(h)

			# Split the projections
			i,f,z,o = proj_t.chunk(4, dim=-1)

			# Activations
			z = torch.tanh(z)
			o = torch.sigmoid(o)
			f = F.logsigmoid(f) + m
			m = torch.maximum(f, i) if torch.is_tensor(m) else i
			i = torch.exp(i - m)
			f = torch.exp(f - m)

			# Gating
			c = f*c + i*z
			n = f*n + i
			h = o*(c/n)

			# Save the output
			output.append(h)
		output = torch.stack(output, dim=1)
		output = self.headwise_norm(self.dropout(output))
		output = output.view(B,L,D) # Merge heads
		return output,(h,c)

class FeedForward(nn.Module):
	def __init__(self, d_model, dim_feedforward):
		super().__init__()
		self.linear1 = nn.Linear(d_model, 2*dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

	def forward(self, x):
		x1,x2 = self.linear1(x).chunk(2, dim=-1)
		x2 = F.gelu(x2)
		x = self.linear2(x1*x2)
		return x