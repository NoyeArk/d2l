import torch
import torchvision
from IPython import display
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms


# 模型超参数
batch_size = 256
num_inputs = 784
num_outputs = 10


class Accumulator:
	"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n
	
	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]

	def reset(self):
		self.data = [0.0] * len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def get_fashion_mnist_labels(labels):
	"""构造数据集的文本标签"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
	"""可视化图片"""
	figsize = (num_cols * scale, num_rows * scale)
	_, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
	axes = axes.flatten()
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		if torch.is_tensor(img):
			ax.imshow(img.numpy())
		else:
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
	return axes


def get_dataloadere_workders():
	"""使用4个进程来读取数据"""
	return 4


def load_data_fashion_mnist(batch_size, resize=None):
	"""下载Fashion-MNIST数据集, 然后将其加载到内存中"""
	trans = [transforms.ToTensor()]
	if resize:
		trans.append(transforms.Resize(resize))
	trans = transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True
	)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True
	)
	return (
		data.DataLoader(mnist_train, batch_size, shuffle=True, num_works=get_dataloadere_workders()),
		data.DataLoader(mnist_test, batch_size, shuffle=True, num_works=get_dataloadere_workders())
	)


def softmax(X):
	X_exp = torch.exp(X_exp)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition  # 使用广播机制


def net(X):
	"""定义模型"""
	return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)


def cross_entropy(y_hat, y):
	"""定义损失函数"""
	return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
	"""计算预测正确的数量"""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
	"""计算在指定数据集上模型的精度"""
	if isinstance(net, torch.nn.Module):
		net.eval()  # 评估模式
	metric = Accumulator(2)
	with torch.no_grad():
		for X, y in data_iter:
			metric.add(accuracy(net(X), y), y.numel())
	return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
	"""训练模型一个迭代周期"""
	if isinstance(net, torch.nn.Module):
		net.train()
	# 训练损失总和、训练准确度总和、样本数
	metric = Accumulator(3)
	for X, y in train_iter:
		# 计算梯度并更新参数
		y_hat = net(X)
		l = loss(y_hat, y)
		if isinstance(updater, torch.optim.Optimizer):
			# 使用Pytorch内置的优化器和损失函数
			updater.zero_grad()
			l.mean().backward()
			updater.step()
		else:
			# 使用定制的优化器和损失函数
			l.sum().backward()
			updater(X.shape[0])
		metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
	# 返回训练损失和训练精度
	return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
	"""训练模型"""
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
	train_loss, train_acc = train_metrics
	assert train_loss < 0.5, train_loss
	assert train_acc <= 1 and train_acc > 0.7, train_acc
	assert test_acc <= 1 and test_acc > 0.7, test_acc


if __name__ == "__main__":
	# 下载数据集
	trans = transforms.ToTensor()  # 通过ToTensor将图像数据从PIL类型变成32位浮点数格式
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True
	)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True 
	)

	# 查看数据集的大小
	print(f'训练集的大小为{len(mnist_train)} 测试集的大小为{len(mnist_test)}')

	# 构造数据集
	X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
	show_images(X.reshape(18, 28, 28), 3, 6, titles=get_fashion_mnist_labels(y))

	# 读取小批量
	train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloadere_workders())

	# 查看读取数据需要多少时间
	time = d2l.Timer()
	for X, y in train_iter:
		continue
	print(f'{time.stop():.2f} sec')

	# 初始化模型参数
	W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
	b = torch.zeros(num_outputs, requires_grad=True)

	# accuracy(y_hat, y) / len(y)

	# 训练
	
