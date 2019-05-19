from abc import ABC, abstractmethod
import torch.nn as nn


class GroupBlock(ABC, nn.Module):

	@property
	def features(self):
		return self.features

	def __init__(self,features):
		super().__init__()
		self._features = features
		
	
	@abstractmethod
	def get_merge_target(self,*kwargs):
		...
