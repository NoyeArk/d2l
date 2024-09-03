import torch
import pandas as pd
from model import Model


if __name__ == '__main__':
    net = torch.load("../model/model7_0.pth")

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
