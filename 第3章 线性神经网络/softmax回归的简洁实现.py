import torch
import torch.nn as nn
import d2l.torch as d2l


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


if __name__ == '__main__':
    # 超参数
    lr = 0.005
    epochs = 10
    batch_size = 256

    # 加载数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    loss = nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(net.parameters(), lr)

    for epoch in range(epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            _loss = loss(y_hat, y)
            optim.zero_grad()
            _loss.mean().backward()
            optim.step()
        metric.add(float(_loss.sum()), accuracy(y_hat, y), y.numel())

        print(f'epoch:{epoch}, loss:{metric[0] / metric[2]}, acc:{metric[1] / metric[2]}')

        # 计算在训练集上的模型的精度
        metric = d2l.Accumulator(2)
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                metric.add(accuracy(net(X), y), y.numel())

            print(f'epoch:{epoch}, acc:{metric[0] / metric[1]}')
