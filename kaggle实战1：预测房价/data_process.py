import torch
import numpy as np
import pandas as pd


if __name__ == "__main__":
	train_data = pd.read_csv("data/train.csv")
	test_data = pd.read_csv("data/test.csv")

	"""查看数据集"""
	print(f'训练集的形状：{train_data.shape}')
	print(f'测试集的形状：{test_data.shape}')

	"""查看前四个和后两个特征"""
	print("查看前4个和后4个特征:", train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

	"""将训练集和测试集组合在一起"""
	all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

	"""数据预处理"""
	# 筛选出所有数值类型
	numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
	all_features[numeric_features] = all_features[numeric_features].apply(
		lambda x : (x - x.mean()) / (x.std())
	)

	# 在标准化数据之后 所有均值消失 可以将缺失值设置为0
	all_features[numeric_features] = all_features[numeric_features].fillna(0)

	# 处理离散值 使用独热编码进行替换
	# all_features = pd.get_dummies(all_features, dummy_na=True)

	print(f'处理之后数据集的形状：{all_features.shape}')

	"""将数据从pandas格式中提取numpy格式"""
	n_train = train_data.shape[0]
	train_features = torch.tensor(all_features[numeric_features][:n_train].values, dtype=torch.float32)
	test_features = torch.tensor(all_features[numeric_features][n_train:].values, dtype=torch.float32)
	train_labels = torch.tensor(train_data.SoldPrice.values.reshape(-1, 1), dtype=torch.float32)