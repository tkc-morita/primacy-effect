# coding: utf-8

import torch
import torch.nn.functional as F

class _Base(object):
	def __init__(self, dummy_datasize=512):
		self.dummy_datasize = dummy_datasize
	
	def __len__(self):
		return self.dummy_datasize

class NonRepeatRandomSequence(_Base):
	collate_fn = None
	@staticmethod
	def holdout_data(vocab_size, length, num_held_out=0,):
		assert vocab_size>=length, 'vocab_size must be at least length.'
		# NOTE: torch.prod more easily explodes than Python's builtin math.
		log_possible_patterns = torch.arange(1,vocab_size+1)[-length:].log().sum().item() # Factorial
		log_possible_patterns -= torch.arange(1,length+1).log().sum().item() # Combination
		assert log_possible_patterns>torch.tensor(num_held_out).log().item(), 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
		held_out = torch.randperm(vocab_size)[None,:length].sort(dim=-1)[0] # Sort for ignoring orders.
		while held_out.size(0)<num_held_out:
			candidate = torch.randperm(vocab_size)[None,:length].sort(dim=-1)[0] # Sort for ignoring orders.
			if (candidate!=held_out).any(dim=-1).all(dim=0).item(): # check duplication
				held_out = torch.cat([held_out,candidate], dim=0)
		return held_out

	def __init__(self, vocab_size, length, held_out=None, **kwargs):
		super().__init__(**kwargs)
		assert vocab_size>=length, 'vocab_size must be at least length.'
		self.vocab_size = vocab_size
		self.length = length
		self.held_out = held_out

	def __getitem__(self, ix):
		while True: # Rejection sampling
			sequence = torch.randperm(self.vocab_size)[:self.length]
			if self.held_out is None or (sequence.sort(dim=-1)[0]!=self.held_out).any(dim=-1).all(dim=0).item():
				break
		return sequence


class AssociativeRecallDataset(_Base):
	collate_fn = None
	@staticmethod
	def holdout_data(target_vocab_size, length, query_vocab_size=None, num_held_out=0,):
		if query_vocab_size is None:
			query_vocab_size = target_vocab_size
		assert length%2==0, 'length must be even.'
		assert min(target_vocab_size,query_vocab_size)*2>=length, '2*vocab_size must be at least length.'
		held_out = AssociativeRecallDataset._sample(query_vocab_size, target_vocab_size, length).unsqueeze(0)
		while held_out.size(0)<num_held_out:
			candidate = AssociativeRecallDataset._sample(query_vocab_size, target_vocab_size, length).unsqueeze(0)
			if (candidate!=held_out).any(dim=(-2,-1)).all(dim=0).item(): # check duplication
				held_out = torch.cat([held_out,candidate], dim=0)
		return held_out
	
	@staticmethod
	def _sample(query_vocab_size, target_vocab_size, length):
		return torch.stack([torch.randperm(query_vocab_size)[:length//2]+target_vocab_size, # starting from target_vocab_size
							torch.randperm(target_vocab_size)[:length//2]],
							dim=-1)

	def __init__(self, target_vocab_size, length, query_vocab_size=None, held_out=None, **kwargs):
		super().__init__(**kwargs)
		if query_vocab_size is None:
			query_vocab_size = target_vocab_size
		assert length%2==0, 'length must be even.'
		assert min(target_vocab_size,query_vocab_size)*2>=length, '2*vocab_size must be at least length.'
		self.query_vocab_size = query_vocab_size
		self.target_vocab_size = target_vocab_size
		self.length = length
		self.held_out = held_out

	def __getitem__(self, ix):
		while True: # Rejection sampling
			sequence = self._sample(self.query_vocab_size, self.target_vocab_size, self.length)
			if self.held_out is None or (sequence!=self.held_out).any(dim=(-2,-1)).all(dim=0).item():
				break
		return sequence