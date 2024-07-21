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
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition


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

