import torch
from torch import nn
from torch.utils import data # type: ignore


def synthetic_data(w, b, num_examples):
	"""生成数据集"""
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.001, y.shape)
	return X, y.reshape(-1, 1)


epoches = 3
batch_size = 10


if __name__ == "__main__":
	true_w = torch.tensor([2, -3.4])
	true_b = 4.2
	
	# 构造数据集
	features, labels = synthetic_data(true_w, true_b, 1000)

	# 读取数据集
	dataset = data.TensorDataset(*(features, labels))
	data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

	# 定义模型
	net = nn.Sequential(nn.Linear(2, 1))

	# 初始化模型参数（可以没有）
	net[0].weight.data.normal_(0, 0.01)
	net[0].bias.data.fill_(0)

	# 定义损失函数
	loss = nn.HuberLoss(reduction='sum', delta=0.1)
	# loss = nn.MSELoss()

	# 定义优化算法
	trainer = torch.optim.SGD(net.parameters(), lr=0.03)

	# 训练神经网络
	for epoch in range(epoches):
		for X, y in data_iter:
			y_pred = net(X)
			l = loss(y_pred, y)
			# 已经计算完梯度了 所以后面不需要记录梯度
			trainer.zero_grad()
			# 反向传播计算梯度
			l.backward()
			# 更新模型参数
			trainer.step()
			
		l = loss(net(features), labels)
		print(f'epoch {epoch + 1}, loss {l:f}')
	
	# 查看训练结果
	w = net[0].weight.data
	print('w的估计误差：', true_w - w.reshape(true_w.shape))
	b = net[0].bias.data
	print('b的估计误差：', true_b - b)
