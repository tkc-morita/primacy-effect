# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .headwise_linear import HeadWiseLinear

class mLSTMBlock(nn.Module):
	def __init__(self, d_model, nhead, dropout, num_blocks):
		super().__init__()
		self.ln = nn.LayerNorm(d_model)
		self.linear1 = nn.Linear(d_model, 4*d_model)
		nn.init.normal_(self.linear1.weight, std=math.sqrt(2/5/d_model))
		self.linear2 = nn.Linear(2*d_model, d_model)
		nn.init.normal_(self.linear2.weight, std=2/num_blocks/math.sqrt(d_model))
		self.mlstm = mLSTMCore(2*d_model, nhead)
		self.causal_conv = CausalConv(2*d_model)
		self.register_parameter('learnable_skip',
			nn.Parameter(torch.ones(2*d_model), requires_grad=True))
		self.dropout = nn.Dropout(dropout)

	def forward(self, input):
		x1,x2 = self.linear1(self.ln(input)).chunk(2,-1)

		qk = self.causal_conv(x1)
		qk = F.silu(qk)
		x1 = self.mlstm(qk, x1) + self.learnable_skip*qk

		x = x1 * F.silu(x2)
		output = self.dropout(self.linear2(x)) + input
		return output

class mLSTMCore(nn.Module):
	def __init__(self, d_wide, nhead):
		super().__init__()
		dim_per_head = d_wide//nhead
		self.nhead = nhead
		self.proj_qk = HeadWiseLinear(d_wide, 2*d_wide, nhead)
		self.proj_v = HeadWiseLinear(d_wide, d_wide, nhead)
		self.proj_if = nn.Linear(3*d_wide, 2*nhead)
		nn.init.zeros_(self.proj_if.weight)
		self._init_forget_bias()
		# self.proj_o = nn.Linear(3*d_wide, dim_per_head)
		self.headwise_norm = nn.LayerNorm(dim_per_head, bias=False) # NOTE: Following official implementation

	def _init_forget_bias(self):
		# NOTE: forget gate needs special initialization.
		f_init = torch.linspace(3.0, 6.0, self.nhead).view(-1)
		# NOTE: And input gate here follows the normal distribution
		i_init = torch.randn_like(f_init)*0.1
		init_vals = torch.cat([i_init,f_init], dim=0)
		with torch.no_grad():
			self.proj_if.bias.copy_(init_vals)

	def forward(self, qk, v):
		B,L,D = qk.size()
		qk = self.proj_qk(qk) # -> B x L x H x 2*D/H
		q,k = qk.chunk(2, dim=-1)
		v = self.proj_v(v) # -> B x L x H x D/H
		qkv = torch.cat([qk,v], dim=-1).view(B,L,3*D)
		i,f = self.proj_if(qkv # -> B x L x 2*H
				).chunk(2, dim=-1) # -> B x L x H each
		# o = self.proj_o(qkv) # NOTE: output gate is externalized as the silu-activated multiplication
		# o = F.sigmoid(o)
		i = torch.exp(i)

		f = F.logsigmoid(f)
		f = f.cumsum(dim=1)
		f = f.view(B,L,1,-1) - f.view(B,1,L,-1)
		mask = torch.ones(L,L, device=f.device).triu(diagonal=1).bool().view(1,L,L,1)
		f = f.masked_fill(mask, -torch.inf)

		d = f + i.unsqueeze(1)
		m = d.amax(dim=-2, keepdim=True)
		d = torch.exp(d-m)

		k = k / math.sqrt(k.size(-1))
		c = torch.einsum('bohd,bihd->boih',q,k)*d
		b = c.sum(dim=-2,keepdim=True)
		n = torch.maximum(b.abs(), torch.exp(-m))
		c = c / n

		h = torch.einsum('boih,bihd->bohd',c,v)
		# h = self.headwise_norm(o*h).view(B,L,D)
		h = self.headwise_norm(h).view(B,L,D)
		return h

class CausalConv(nn.Conv1d):
	def __init__(self, channels, kernel_size=4):
		super().__init__(channels, channels, kernel_size, padding=0)
		self.left_pad = kernel_size-1

	def forward(self, input):
		input = input.permute(1,2,0) # LxBxD -> BxDxL
		input = F.pad(input, (self.left_pad,0))
		output = super().forward(input)
		output = output.permute(2,0,1).contiguous() # BxDxL -> LxBxD
		return output