import torch.nn as nn


class Model(nn.Module):
	def __init__(self, dim=18):
		super(Model, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

	def forward(self, x):
		return self.net(x)
