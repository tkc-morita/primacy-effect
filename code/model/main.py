# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from .ssm import S4,Mamba
from .xlstm import xLSTM

class Memorizer(nn.Module):
	def __init__(self, model_name, vocab_size, hidden_size, output_size=None, extra_input_symbols=0,
					encode_phase=False, **kwargs):
		super().__init__()
		self.embed_input = nn.Embedding(vocab_size+extra_input_symbols, hidden_size)
		if encode_phase:
			self.embed_phase = nn.Embedding(2, hidden_size)
			self.merge_embeddings = nn.Linear(2*hidden_size, hidden_size)
		else:
			self.embed_phase = None
		if model_name=='Transformer':
			self.seq_model = Transformer(d_model=hidden_size, batch_first=True, **kwargs)
		elif model_name=='S4':
			self.seq_model = S4(d_model=hidden_size, **kwargs)
		elif model_name=='Mamba':
			self.seq_model = Mamba(d_model=hidden_size, **kwargs)
		elif model_name=='xLSTM':
			self.seq_model = xLSTM(d_model=hidden_size, **kwargs)
		elif model_name in ['RNN','GRU','LSTM']:
			self.seq_model = getattr(nn, model_name)(input_size=hidden_size, hidden_size=hidden_size,
													batch_first=True, **kwargs)
		else:
			raise ValueError('Unsupported model: {}'.format(model_name))
		self.to_logits = nn.Linear(hidden_size, vocab_size if output_size is None else output_size)
			
	def forward(self, input, phase=None, target=None, loss_func=None):
		"""
		input: batch_size x length
		"""
		input = self.embed_input(input)
		if not self.embed_phase is None:
			phase = self.embed_phase(phase)
			input = torch.cat([input,phase], dim=-1)
			input = self.merge_embeddings(input)
		output = self.seq_model(input)
		if not isinstance(output, torch.Tensor):
			output = output[0] # Strip off the state od RNNs
		output = self.to_logits(output)
		if loss_func is None:
			return output
		else:
			return loss_func(output, target)
		