from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

def GroupBlock(ABC,nn.Module):
	
	@property
	def features(self):
		return self.features

	def __init__(self,features):
		super().__init__()
		self._features = features
		
	
	@abstractmethod
	def get_merge_target(self,*kwargs):
		...
	
	