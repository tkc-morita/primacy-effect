# coding: utf-8

from .mamba import Mamba as _Mamba
from .mamba import MambaConfig

class Mamba(_Mamba):
	def __init__(self, *args, num_layers=1, **kwargs):
		super().__init__(MambaConfig(*args, n_layers=num_layers, **kwargs))