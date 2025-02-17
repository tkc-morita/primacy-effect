#!/bin/bash

save_dir=/somewhere/to/save/results
model_name=HiPPO # Choose one of {HiPPO, S4, LSTM}. HiPPO freezes the state and input matrices (A and B) while S4 optimizes them.
seed=01 # Random seed. Choose one from 01-10


hidden_size=256 # #channels

# Experiment params
seq_length=128 # Length of study items.
vocab_size=4096 # Vocaburary size
num_held_out=1024 # #COMBINATIONS of test study items.

# SSM params
ssm_init="legs" # Choose one of {legs, lagt, fout}. 
dt_transform=exp # you can also test "softplus"
ssm_mode=nplr # you can also test "diag"
disc=bilinear # you can also test "zoh"
real_transform=exp
dt_min=0.001 # Minimum value of delta t in log-uniform initialization.
dt_max=0.1 # Maximum value of delta t in log-uniform initialization.

## Summarize the 
ssm_config=`[ ! -z "$ssm_mode" ] && echo "--ssm_mode ${ssm_mode} "``[ ! -z "$ssm_init" ] && echo "--ssm_init $ssm_init "``[ ! -z "$disc" ] && echo "--disc ${disc} "``[ ! -z "$dt_transform" ] && echo "--dt_transform ${dt_transform}"``[ ! -z "$freeze_B" ] && echo "--freeze_B "``[ ! -z "$real_ssm" ] && echo "--real_ssm "``[ ! -z "$dt_min" ] && echo "--dt_min ${dt_min}"``[ ! -z "$dt_max" ] && echo "--dt_max ${dt_max}"`

# Model params
num_layers=1
dropout=0.0

# Training params.
train_iterations=300000
batch_size=512
saving_interval=5000
num_workers=4
warmup_iters=1000
learning_rate=1e-3

device=cuda

python code/train_verification.py $vocab_size $seq_length $save_dir \
	--model_name $model_name \
	--hidden_size $hidden_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$ssm_config \
	--num_held_out $num_held_out