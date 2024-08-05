import torch
import pandas as pd
import torch.nn as nn
from model import Model
from d2l import torch as d2l


"""
训练一个神经网络需要准备的组件：
1. 实例化的模型
2. 优化器
3. 训练集和测试集
4. 损失函数
"""

"""训练超参数定义"""
k = 10
lr = 0.003
epochs = 30
batch_size = 32


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def log_rmse(net, loss, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse



def train(net, train_features, train_labels, test_features, test_labels):
    train_loss, test_loss = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    loss = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, y in train_iter:
            # 清除上次计算的梯度，准备新一轮的计算
            optim.zero_grad()
            # 计算模型输出
            y_hat = net(X)
            # 计算和目标值之间的损失
            l = loss(y_hat, y)
            # 反向传播
            optim.step()
        
        train_loss.append(log_rmse(net, loss, train_features, train_labels))
        if  test_labels is not None:
            test_loss.append(log_rmse(net, loss, test_features, test_labels))

    return train_loss, test_loss


def k_fold(net, X_train, y_train):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_loss, valid_loss = train(net, *data)
        train_l_sum += train_loss[-1]
        valid_l_sum += valid_loss[-1]
        if i == 0:
            d2l.plot(list(range(1, epochs + 1)), [train_loss, valid_loss], 
                    xlabel='epoch', ylabel='rmse', xlim=[1, epochs],
                    legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_loss[-1]):f}, '
                f'验证log rmse{float(valid_loss[-1]):f}')
        
    return train_l_sum / k, valid_l_sum / k


if __name__ == "__main__":
	net = Model()

	# 准备数据集
	train_data = pd.read_csv("data/processed_train.csv")
	test_data = pd.read_csv("data/processed_test.csv")

	train_features = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
	train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)
	test_features = torch.tensor(test_data.values, dtype=torch.float32)

	print(f'标签是{train_labels}')

	train_l, valid_l = k_fold(net, train_features, train_labels)
	print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
		f'平均验证log rmse: {float(valid_l):f}')

