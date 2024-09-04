"""
线性回归的简洁实现
1. 生成数据
2. 构建模型
3. 选择损失函数
4. 选择优化器
5. 进行训练
    5.1 取一个批次的数据
    5.2 送入模型进行预测
    5.3 计算损失
    5.4 梯度清零
    5.5 计算梯度
    5.6 反向传播
"""

import torch
import torch.nn as nn
from torch.utils import data


def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.001, y.shape)
    return X, y.reshape(-1, 1)


if __name__ == '__main__':
    # 训练超参数
    lr = 0.03
    epochs = 3
    batch_size = 12

    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor(4.2)

    # 构造数据集
    features, labels = generate_data(true_w, true_b, 1000)

    # 读取数据集
    dataset = data.TensorDataset(*(features, labels))
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化算法
    optim = torch.optim.SGD(net.parameters(), lr)

    # 训练模型
    for epoch in range(epochs):
        for X, y in data_iter:
            y_hat = net(X)
            _loss = loss(y_hat, y)
            optim.zero_grad()
            _loss.backward()
            optim.step()

        _loss = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {_loss:f}')

    # 查看训练结果
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
