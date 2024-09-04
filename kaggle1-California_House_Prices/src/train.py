import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from model import Model
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

"""
训练一个神经网络需要准备的组件：
1. 实例化的模型
2. 优化器
3. 训练集和测试集
4. 损失函数
"""

"""训练超参数定义"""
lr = 0.003
epochs = 30
batch_size = 48
num_gpu = 1


def log_mse(net, loss, features, labels):
    """

    """
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse


def train(net, train_features, train_labels, test_features, test_labels) -> (list, list):
    """
    使用传入的数据对net进行epochs轮训练.

    Args:
        net: 实例化的模型
        train_features: 训练集数据
        train_labels: 训练集标签
        test_features: 测试集数据
        test_labels: 测试集标签

    Returns:
        `list[int]`: 不同epoch的训练集损失
        `list[int]`: 不同epoch的验证集损失

    """
    net.train()

    train_loss, test_loss = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    loss = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), colour='blue'):
        for X, y in train_iter:
            # 清除上次计算的梯度，准备新一轮的计算
            optim.zero_grad()
            # 计算模型输出
            y_hat = net(X)
            # 计算和目标值之间的损失
            _loss = loss(y_hat, y)
            # 计算梯度
            _loss.backward()
            # 反向传播
            optim.step()

        _loss = log_mse(net, loss, train_features, train_labels)
        writer[0].add_scalar(tag="train_loss", scalar_value=_loss, global_step=epoch)
        train_loss.append(_loss)
        if test_labels is not None:
            writer[1].add_scalar(tag="valid_loss", scalar_value=_loss, global_step=epoch)
            test_loss.append(log_mse(net, loss, test_features, test_labels))

    writer[0].close()
    writer[1].close()
    return train_loss, test_loss


def k_fold(net, X_train, y_train):
    # 训练集和验证集的损失
    train_l_sum, valid_l_sum = 0, 0

    # 多少折就需要多少个writer
    writers = {'train': [SummaryWriter(log_dir=f'../runs/train/kfold_{i}') for i in range(k)],
               'valid': [SummaryWriter(log_dir=f'../runs/valid/kfold_{i}') for i in range(k)]}

    for i in range(k):
        # 得到一折的数据
        data = get_k_fold_data(k, i, X_train, y_train)
        # 对这些数据进行训练
        train_loss, valid_loss = train(net, [writers['train'][i], writers['valid'][i]], *data)
        train_l_sum += train_loss[-1]
        valid_l_sum += valid_loss[-1]

        print(f'折{i + 1}，训练log mse:{float(train_loss[-1]):f}, 'f'验证log mse:{float(valid_loss[-1]):f}')

    return train_l_sum / k, valid_l_sum / k


def pred_and_save_model(net, model_name):
    test_data = pd.read_csv("../data/processed_test.csv")
    row_test_data = pd.read_csv("../data/test.csv")
    test_features = torch.tensor(test_data.values, dtype=torch.float32)
    test_features = test_features.to("cuda")

    # 对测试集进行预测
    net.eval()
    pred = net(test_features).detach().cpu().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([row_test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

    # 保存模型
    torch.save(net, f"../model/{model_name}.pth")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    net = Model()
    net = net.to(device)

    # 读取数据集
    train_data = pd.read_csv("../data/processed_train.csv")

    # 将数据由pandas转为tensor
    train_features = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)

    train_features = train_features.to("cuda:0")
    train_labels = train_labels.to("cuda:0")

    train_x, train_y, valid_x, valid_y = spli

    train_l, valid_l = train(net, train_features, train_labels)
    print(f'训练log mse: {float(train_l):f}, 'f'验证log mse: {float(valid_l):f}')

    pred_and_save_model(net, "model7")
