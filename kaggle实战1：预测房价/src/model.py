import torch.nn as nn


class Model(nn.Module):
	def __init__(self, dim=18):
		super(Model, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, 32),
			nn.Linear(32, 16),
			nn.Linear(16, 8),
			nn.Linear(8, 1)
		)

	def forward(self, x):
		return self.net(x)
