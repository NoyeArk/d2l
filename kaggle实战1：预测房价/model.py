import torch.nn as nn


class Model(nn.Module):
	def __init__(self, hidden_dim=64):
		super().__init__()
		self.w = nn.Linear(20, hidden_dim)

	def forward(self, x):
		y = self.w(x)
		return y