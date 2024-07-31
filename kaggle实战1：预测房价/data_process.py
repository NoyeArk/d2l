import pandas as pd


if __name__ == "__main__":
	train_data = pd.read_csv("")
	test_data = pd.read_csv("")

	"""查看数据集"""
	print(f'训练集的形状：{train_data.data}')
	print(f'测试集的形状：{test_data.data}')

	"""查看前四个和后两个特征"""
	print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

	"""将训练集和测试集组合在一起"""
	all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

	"""数据预处理"""
	# 筛选出所有数值类型
	numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index


