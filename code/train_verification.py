# coding: utf-8

import os,argparse,math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.training_template import Learner as _Learner
from utils.logging import get_logger
from data.dataset import NonRepeatRandomSequence

class Learner(_Learner):
	@staticmethod
	def _permutation(sequence, perm_order):
		permuted = sequence.gather(index=perm_order.to(sequence.device), dim=-1)
		return permuted
	
	@staticmethod
	def _inv_permutation(permuted, perm_order):
		inv_idx = torch.scatter(perm_order, dim=-1, index=perm_order,
							src=torch.arange(perm_order.size(-1), device=perm_order.device
											).unsqueeze(0).expand_as(perm_order))
		inversed = permuted.gather(index=inv_idx, dim=-1)
		return inversed
		
	def _replace(self, sequence, unmask=None):
		if unmask is None:
			unmask = torch.randint_like(sequence, 2, dtype=torch.bool)
		in_sequence = self._batch_wise_bincount(sequence, self.checkpoint['modules']['model']['init_args']['vocab_size'])
		# NOTE: torch.multinomial(replacement=False) does NOT raise an error
		#       even when # of non-zero weights is smaller than # of samples.
		complements = torch.multinomial((in_sequence==0).float(), sequence.size(-1), replacement=False)
		replaced = sequence.where(unmask, complements)
		return unmask,replaced
	
	@staticmethod
	def _batch_wise_bincount(x, minlength=None):
		# NOTE: Memory efficient implementation proposed in https://discuss.pytorch.org/t/count-number-occurrence-of-value-per-row/137061/4
		if minlength is None:
			minlength = x.max() + 1
		# Step 1; Add the offset
		B,L = x.size()
		x = x + minlength*torch.arange(B, device=x.device).unsqueeze(1)
		# Step 2: Binbount in the flattened vector
		counts = torch.bincount(x.flatten(), minlength=minlength*B).view(B,minlength)
		return counts

	def _reshuffle(self, sequence):
		B,L = sequence.size()
		query_order = torch.stack([torch.randperm(L) for _ in range(B)],dim=0).to(sequence.device)
		return query_order,self._permutation(sequence, query_order)
	
	def train_per_iteration(self, sequence, records, iteration):
		self.optimizer.zero_grad()
		sequence = sequence.to(self.device)

		query_order,query = self._reshuffle(sequence)
		target,query = self._replace(query)

		input = torch.cat([sequence,query], dim=1)
		phase = torch.cat([torch.zeros_like(sequence),torch.ones_like(query)], dim=1) \
				if self.checkpoint['modules']['model']['init_args'].get('encode_phase', False) \
				else None

		def loss_func(logits, target):
			"""
			logits: batch_size x 2*length
			target: batch_size x length (long int) x 1
			"""
			logits = logits[:,-target.size(1):].squeeze(-1) # Or simply logits[:,-1,:]
			loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none').mean(dim=-1)

			# Stats for logging below
			accuracy = ((logits>0)==target).float().mean(dim=-1)
			return loss,accuracy

		loss,accuracy \
			= self.model(input, target=target, loss_func=loss_func, phase=phase)
		loss = loss.mean()

		self.update_records(records, 'loss', loss.item())
		self.update_records(records, 'accuracy', accuracy.mean().item())

		loss.backward()
		clip_grad_norm_(self.get_parameters(), 1.0)
		self.optimizer.step()
		self.scheduler.step(iteration)
		return records

	def log_training_stats(self, records, saving_interval):
		self.logger.info('Binary cross entropy loss (perplexity): {:0.6f}'.format(math.exp(records['loss']/saving_interval)))
		self.logger.info('Accuracy: {:0.6f}'.format(records['accuracy']/saving_interval))

	def test(self, whole_data, *args):
		save_path = os.path.join(self.save_dir, 'test.csv')
		num_seqs,L = whole_data.size()
		accuracy_per_time=dict(
							uniform=torch.zeros((L+1,L)),
							half=torch.zeros((L+1,L)),
							)
		self.model.eval()
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		is_even_time = (torch.arange(L) % 2 == 0).unsqueeze(0)
		encode_phase = self.checkpoint['modules']['model']['init_args'].get('encode_phase', False)
		for sequence in whole_data.unbind(dim=0):
			# Step 1: Shuffle input to remove potential correlation b/w vocab vs. order.
			sequence = self._permutation(sequence, torch.randperm(L))
			# Step 2: Shuffle output query
			query_order = torch.randperm(L)
			# Step 3: Shift output query to exhaust the pairwise combinations of the input and output positions.
			query_order = torch.stack([query_order]+[query_order.roll(shifts, dims=-1) for shifts in range(1,L)], dim=0)

			sequence = sequence.unsqueeze(0)
			_,neg_query = self._replace(sequence, unmask=torch.zeros_like(sequence, dtype=torch.bool))
			pos_query = self._permutation(sequence.expand_as(query_order), query_order)
			even_pos_query = pos_query.where(is_even_time, neg_query)
			odd_pos_query = pos_query.where(~is_even_time, neg_query)
			query = torch.cat([pos_query,neg_query,even_pos_query,odd_pos_query])
			input = torch.cat([sequence.expand_as(query),query], dim=-1)
			phase = torch.cat([torch.zeros_like(query),torch.ones_like(query)], dim=1).to(self.device) \
						if encode_phase else None

			logits = self.model(input.to(self.device), phase=phase)
			logits = logits[:,-L:].squeeze(-1).cpu() # Bx2L -> BxL

			pos_logits = logits[:L,:]
			accuracy_per_time['uniform'].scatter_add_(dim=0, index=query_order, src=(pos_logits>0).float())

			neg_logits = logits[L,:]
			accuracy_per_time['uniform'][-1,:] += (neg_logits<=0).float()

			even_pos_logits,odd_pos_logits = logits[L+1:,:].chunk(2, dim=0)
			is_correct_half_pos = even_pos_logits.where(is_even_time, odd_pos_logits)>0
			accuracy_per_time['half'].scatter_add_(dim=0, index=query_order, src=is_correct_half_pos.float())

			is_correct_half_neg = even_pos_logits.where(~is_even_time, odd_pos_logits)<=0
			accuracy_per_time['half'][-1,:] += is_correct_half_neg.float().mean(dim=0)

		def format_df(query_type):
			sub_df = pd.DataFrame((accuracy_per_time[query_type]/num_seqs).numpy())
			sub_df['input_time'] = sub_df.index.map(lambda x: -1 if x==L else x)
			sub_df = sub_df.melt(id_vars='input_time', var_name='output_time', value_name='accuracy')
			sub_df['query_type'] = query_type
			return sub_df
		df = pd.concat([format_df('uniform'), format_df('half')], axis=0)
		df.to_csv(save_path, index=False)




if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_size', type=int, help='Vocabulary size.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_held_out', type=int, default=0, help='# of random sequences to be held out for testing.')

	parser.add_argument('--encode_inout_phase', action='store_true', help='Provide the model with an auxiliary input sequence binary-encoding the input vs. output phase.')

	parser.add_argument('--model_name', type=str, required=True, choices=['RNN','GRU','LSTM','xLSTM','Transformer','S4','HiPPO','Mamba'], help='Type of sequence model.')
	parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN/Transformer/SSM.')
	parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN/Transformer/SSM.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN/Transformer/SSM.')

	# Transformer/xLSTM
	parser.add_argument('--nhead', type=int, default=None, help='# of attention heads of Transformer.')
	parser.add_argument('--dim_feedforward', type=int, default=None, help='Dimentionality of FF layers in Transformer. 4*hidden_size by default.')

	# Transformer/S4
	parser.add_argument('--activation', type=str, default='gelu', choices=['gelu','relu'], help='Activation function for Transformer/S4')

	# S4
	parser.add_argument('--ssm_mode', type=str, default='nplr', choices=['dplr','diag','nplr'], help='Type of S4 state matrix. NOTE: "dplr" is equivalent to "nplr" unless "init" option (not yet implemented here) is headed by "dplr" or "diag". Moreover, they are only different in the implementation of initializations (for building the same parameters).')
	parser.add_argument('--ssm_init', type=str, default='legs', choices=['legs','fout','lagt'], help='Type of the initial HiPPO matrix.')
	parser.add_argument('--d_state', type=int, default=64, help='Dimensionality of state space, or the maximum order of the polynomials.')
	parser.add_argument('--freeze_B', action='store_true', help='Freeze the input2latent projection matrix B, as adopted in S4D.')
	parser.add_argument('--disc' , type=str, default='bilinear', choices=['zoh','bilinear','dss'], help='Type of temporal discretization.')
	parser.add_argument('--real_transform', type=str, default='exp', choices=['exp', 'relu', 'sigmoid', 'softplus'], help='Type of imag-to-real transform.')
	parser.add_argument('--real_ssm', action='store_true', help='Constrain state space to real, reducing the SSM to a EMA.')
	parser.add_argument('--rank', type=int, default=1, help='Rank of the "LR" term of the DPLR.')
	parser.add_argument('--dt_min', type=float, default=0.001, help='Minimum value of time step sizes at log-scaled initialization.')
	parser.add_argument('--dt_max', type=float, default=0.1, help='Maximum value of time step sizes at log-scaled initialization.')
	parser.add_argument('--dt_transform', type=str, default='exp', choices=['exp','softplus'], help='Transformation of the delta time.')

	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer.')
	parser.add_argument('--num_iterations', type=int, default=10000, help='# of training iterations.')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of warm-up iterations.')
	parser.add_argument('--saving_interval', type=int, default=1, help='Intervals of logging of the learning progress.')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of dataloading workers.')

	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')

	args = parser.parse_args()


	os.makedirs(args.save_dir, exist_ok=True)
	logger = get_logger(os.path.join(args.save_dir, 'train.log'))

	logger.info('Learns memory-verification.')
	assert args.vocab_size>args.seq_length*2, "vocab_size must be greater than 2 x seq_length."
	logger.info('Vocabulary size: {}'.format(args.vocab_size))
	logger.info('Sequence length: {}'.format(args.seq_length))

	model_configs = dict()
	init_args=dict(
				vocab_size=args.vocab_size,
				hidden_size=args.hidden_size,
				model_name='S4' if args.model_name in ['HiPPO','Frozen'] else args.model_name,
				num_layers=args.num_layers,
				dropout=args.dropout,
				extra_input_symbols=0,
				output_size=1,
				encode_phase=args.encode_inout_phase,
			)
	if args.model_name=='Transformer':
		init_args.update(dict(
			nhead=8 if args.nhead is None else args.nhead,
			dim_feedforward=4*args.hidden_size if args.dim_feedforward is None
								else args.dim_feedforward,
			activation=args.activation
		))
	elif args.model_name=='xLSTM':
		init_args.update(dict(
			nhead=4 if args.nhead is None else args.nhead,
			dim_feedforward=2*args.hidden_size if args.dim_feedforward is None
								else args.dim_feedforward,
		))
	elif args.model_name in ['S4','HiPPO','Frozen']:
		init_args.update(dict(
			mode=args.ssm_mode,
			init=args.ssm_init,
			d_state=args.d_state,
			lr=dict(B=0.0) if args.freeze_B else None,
			disc=args.disc if args.ssm_mode=='diag' else 'bilinear',
			real_transform=args.real_transform if args.ssm_mode=='diag' else 'exp',
			is_real=args.real_ssm,
			activation=args.activation,
			input_length=args.seq_length*2,
			dt_min=args.dt_min,
			dt_max=args.dt_max
		))
		if args.model_name=='HiPPO': # Freeze A and B parameters of S4 to match the Hippo-LegS.
			init_args.update(dict(
				# init='legs',
				lr=dict(A=0.0,B=0.0), # NOTE: delta_t and C are still learnable.
			))
	elif args.model_name=='Mamba':
		init_args.pop('dropout')
		init_args.update(dict(
			d_state=args.d_state,
		))
	model_configs['model'] = dict(module_name='Memorizer', init_args=init_args)
	optim_config = dict(lr=args.learning_rate, weight_decay=0.0, betas=(0.9,0.98), eps=1e-09)
	scheduler_config = dict(t_initial=args.num_iterations,
								warmup_t=args.warmup_iters,
								warmup_prefix=True, lr_min=0.0)
	learner = Learner(logger, args.save_dir, model_configs, optim_config, scheduler_config,
						device=args.device, seed=args.seed)
	held_out = NonRepeatRandomSequence.holdout_data(args.vocab_size, args.seq_length, args.num_held_out,)
	dataset = NonRepeatRandomSequence(args.vocab_size, args.seq_length, held_out=held_out, dummy_datasize=max(512,args.batch_size))
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)
			# test_batch_size=args.test_batch_size)