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
	def _sample_query(sequence):
		# B,L = sequence.size()
		query_time = torch.rand_like(sequence[:,:-1], dtype=float).argsort(dim=1) # batched randperm
		query = sequence.gather(index=query_time, dim=1)
		target = sequence.gather(index=query_time+1, dim=1)
		return query,target
	
	def train_per_iteration(self, sequence, records, iteration):
		self.optimizer.zero_grad()
		sequence = sequence.to(self.device)

		query,target = self._sample_query(sequence)

		input = torch.cat([sequence,query], dim=1)

		def loss_func(logits, target):
			"""
			logits: batch_size x 2*length-1 x vocab_size
			target: batch_size x length-1 (long int)
			"""
			logits = logits[:,-target.size(1):,:]
			loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)), target.view(-1),
									reduction='none').mean(dim=-1)

			# Stats for logging below
			accuracy = (logits.argmax(dim=-1)==target).float().mean(dim=-1)
			return loss,accuracy

		loss,accuracy \
			= self.model(input, target=target, loss_func=loss_func)
		loss = loss.mean()

		self.update_records(records, 'loss', loss.item())
		self.update_records(records, 'accuracy', accuracy.mean().item())

		loss.backward()
		clip_grad_norm_(self.get_parameters(), 1.0)
		self.optimizer.step()
		self.scheduler.step(iteration)
		return records

	def log_training_stats(self, records, saving_interval):
		self.logger.info('Cross entropy loss (perplexity): {:0.6f}'.format(math.exp(records['loss']/saving_interval)))
		self.logger.info('Accuracy: {:0.6f}'.format(records['accuracy']/saving_interval))

	def test(self, whole_data, **kwargs):
		save_path = os.path.join(self.save_dir, 'test.csv')
		num_seqs,L = whole_data.size()
		accuracy_per_time = torch.zeros(L-1,L-1)
		self.model.eval()
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		for sequence in whole_data.unbind(dim=0):
			sequence = sequence.to(self.device)
			# Step 1: Shuffle input to remove potential correlation b/w vocab vs. order.
			sequence = sequence.gather(index=torch.randperm(L).to(self.device),dim=-1)
			# Step 2: Shuffle output query
			query_order = torch.randperm(L-1).to(self.device)
			# Step 3: Shift output query to exhaust the pairwise combinations of the input and output positions.
			query_order = torch.stack([query_order]+[query_order.roll(shifts, dims=-1) for shifts in range(1,L-1)], dim=0)

			sequence = sequence.unsqueeze(0)

			sequence = sequence.expand(L-1,-1)
			query = sequence.gather(index=query_order, dim=-1)
			target = sequence.gather(index=query_order+1, dim=-1)

			input = torch.cat([sequence,query],dim=1)
			logits = self.model(input)
			logits = logits[:,-target.size(1):,:]
			is_correct = (logits.argmax(dim=-1)==target) # (L-1) x (L-1) matrix


			accuracy_per_time.scatter_add_(dim=0, index=query_order.cpu(), src=is_correct.float().cpu())

		df = pd.DataFrame((accuracy_per_time/num_seqs).numpy())
		df['input_time'] = df.index
		df = df.melt(id_vars='input_time', var_name='output_time', value_name='accuracy')
		df.to_csv(save_path, index=False)




if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_size', type=int, help='Vocabulary size.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_held_out', type=int, default=0, help='# of random sequences to be held out for testing.')

	parser.add_argument('--model_name', type=str, required=True, choices=['RNN','GRU','LSTM','Transformer','S4','HiPPO','Mamba'], help='Type of sequence model.')
	parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN/Transformer/SSM.')
	parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN/Transformer/SSM.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN/Transformer/SSM.')

	# Transformer
	parser.add_argument('--nhead', type=int, default=8, help='# of attention heads of Transformer.')
	parser.add_argument('--dim_feedforward', type=int, default=None, help='Dimentionality of FF layers in Transformer. 4*hidden_size by default.')

	# Transformer/S4
	parser.add_argument('--activation', type=str, default='gelu', choices=['gelu','relu'], help='Activation function for Transformer/S4')

	# S4
	parser.add_argument('--ssm_mode', type=str, default='nplr', choices=['dplr','diag','nplr'], help='Type of S4 state matrix. NOTE: "dplr" is equivalent to "nplr" unless "init" option (not yet implemented here) is headed by "dplr" or "diag". Moreover, they are only different in the implementation of initializations (for building the same parameters).')
	parser.add_argument('--ssm_init', type=str, default='legs', choices=['legs','fout','lagt'], help='Type of the initial HiPPO matrix.')
	parser.add_argument('--d_state', type=int, default=64, help='Dimensionality of state space, or the maximum order of the polynomials.')
	parser.add_argument('--freeze_B', action='store_true', help='Freeze the input2latent projection matrix B, as adopted in S4D.')
	parser.add_argument('--freeze_C', action='store_true', help='Freeze the latent2output projection matrix C.')
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

	logger.info('Learns extended associative recall.')
	logger.info('Vocabulary size: {}'.format(args.vocab_size))
	logger.info('Sequence length: {}'.format(args.seq_length))

	model_configs = dict()
	init_args=dict(
				vocab_size=args.vocab_size,
				hidden_size=args.hidden_size,
				model_name='S4' if args.model_name in ['HiPPO','Frozen'] else args.model_name,
				num_layers=args.num_layers,
				dropout=args.dropout,
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
		lr = dict()
		if args.freeze_B:
			lr['B'] = 0.0
		if args.freeze_C:
			lr['C'] = 0.0
		init_args.update(dict(
			mode=args.ssm_mode,
			init=args.ssm_init,
			d_state=args.d_state,
			lr=lr, #dict(B=0.0) if args.freeze_B else None,
			disc=args.disc if args.ssm_mode=='diag' else 'bilinear',
			real_transform=args.real_transform if args.ssm_mode=='diag' else 'exp',
			is_real=args.real_ssm,
			activation=args.activation,
			input_length=args.seq_length*2,
			dt_min=args.dt_min,
			dt_max=args.dt_max
		))
		if args.model_name=='HiPPO': # Freeze A and B parameters of S4 to match the Hippo-LegS.
			lr.update(A=0.0,B=0.0)
			init_args.update(dict(
				# init='legs',
				lr=lr#dict(A=0.0,B=0.0), # NOTE: delta_t and C are still learnable.
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
	held_out = NonRepeatRandomSequence.holdout_data(args.vocab_size, args.seq_length, args.num_held_out)
	dataset = NonRepeatRandomSequence(args.vocab_size, args.seq_length, held_out=held_out, dummy_datasize=max(512,args.batch_size))
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)