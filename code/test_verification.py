# coding: utf-8

import os,argparse
import pandas as pd
import torch
from utils.logging import get_logger
from train_verification import Learner

class Tester(Learner):
	def __init__(self, logger, checkpoint_path, device='cpu'):
		self.logger = logger
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path=checkpoint_path)

	def __call__(self, save_dir):
		self.logger.info('START OF TEST ON HELD-OUT DATA')
		with torch.no_grad():
			self.save_dir = save_dir
			whole_data = self.checkpoint['held_out_data']
			self.test(whole_data)
		self.logger.info('END OF TEST ON HELD-OUT DATA')

	# def test(self, whole_data, save_path):
	# 	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	# 	# vocab_size = self.checkpoint['modules']['model']['init_args']['vocab_size']
	# 	num_seqs,L = whole_data.size()
	# 	accuracy_per_time=dict(
	# 						uniform=torch.zeros((L+1,L)),
	# 						half=torch.zeros((L+1,L)),
	# 						)
	# 	self.model.eval()
	# 	is_even_time = (torch.arange(L) % 2 == 0).unsqueeze(0)
	# 	for sequence in whole_data.unbind(dim=0):
	# 		# sequence = sequence.to(self.device)
	# 		# Step 1: Shuffle input to remove potential correlation b/w vocab vs. order.
	# 		sequence = self._permutation(sequence, torch.randperm(L))
	# 		# Step 2: Shuffle output query
	# 		query_order = torch.randperm(L)
	# 		# Step 3: Shift output query to exhaust the pairwise combinations of the input and output positions.
	# 		query_order = torch.stack([query_order]+[query_order.roll(shifts, dims=-1) for shifts in range(1,L)], dim=0)

	# 		sequence = sequence.unsqueeze(0)
	# 		_,neg_query = self._replace(sequence, unmask=torch.zeros_like(sequence, dtype=torch.bool))
	# 		pos_query = self._permutation(sequence.expand_as(query_order), query_order)
	# 		even_pos_query = pos_query.where(is_even_time, neg_query)
	# 		odd_pos_query = pos_query.where(~is_even_time, neg_query)
	# 		query = torch.cat([pos_query,neg_query,even_pos_query,odd_pos_query])
	# 		input = torch.cat([sequence.expand_as(query),query], dim=-1)

	# 		logits = self.model(input.to(self.device))
	# 		logits = logits[:,-L:].squeeze(-1).cpu() # Bx2L -> BxL

	# 		# pos_logits,neg_logits,even_pos_logits,odd_pos_logits = logits.chunk(4, dim=0)
	# 		pos_logits = logits[:L,:]
	# 		accuracy_per_time['uniform'].scatter_add_(dim=0, index=query_order, src=(pos_logits>0).float())

	# 		neg_logits = logits[L,:]
	# 		accuracy_per_time['uniform'][-1,:] += (neg_logits<=0).float()

	# 		even_pos_logits,odd_pos_logits = logits[L+1:,:].chunk(2, dim=0)
	# 		is_correct_half_pos = even_pos_logits.where(is_even_time, odd_pos_logits)>0
	# 		accuracy_per_time['half'].scatter_add_(dim=0, index=query_order, src=is_correct_half_pos.float())

	# 		is_correct_half_neg = even_pos_logits.where(~is_even_time, odd_pos_logits)<=0
	# 		accuracy_per_time['half'][-1,:] += is_correct_half_neg.float().mean(dim=0)

	# 	def format_df(query_type):
	# 		sub_df = pd.DataFrame((accuracy_per_time[query_type]/num_seqs).numpy())
	# 		sub_df['input_time'] = sub_df.index.map(lambda x: -1 if x==L else x)
	# 		sub_df = sub_df.melt(id_vars='input_time', var_name='output_time', value_name='accuracy')
	# 		# sub_df.loc[:,'output_time'] += L
	# 		sub_df['query_type'] = query_type
	# 		return sub_df
	# 	df = pd.concat([format_df('uniform'), format_df('half')], axis=0)
	# 	df.to_csv(save_path, index=False)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file containing the trained model.')

	# parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')

	parser.add_argument('--device', type=str, default='cpu', help='Device.')

	args = parser.parse_args()

	# test_log_path = os.path.join(os.path.dirname(args.checkpoint_path), 'test.log')
	logger = get_logger()

	logger.info('Test accuracy of the order-agnostic memory verification.')
	tester = Tester(logger, args.checkpoint_path, device=args.device)

	# test_save_path = os.path.join(os.path.dirname(args.checkpoint_path), 'test.csv')
	tester(os.path.dirname(args.checkpoint_path))