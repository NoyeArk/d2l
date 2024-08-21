import torch.nn as nn


class Model(nn.Module):
	def __init__(self, dim=18, hidden_dim=32):
		super().__init__()
		self.w1 = nn.Linear(dim, hidden_dim)
		self.w2 = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		y = self.w1(x)
		y = self.w2(y)
		return y
