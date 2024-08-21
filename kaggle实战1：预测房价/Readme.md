# Kaggle实战1：预测房价

竞赛地址：https://www.kaggle.com/c/california-house-prices/overview

## 1 打怪升级记录

| 序号 | k-fold | epochs |             学习率              | 优化器  | 网络架构  |            d_model            | 模型大小 |                分数                |
|:--:|:------:|:------:|:----------------------------:|:----:|:-----:|:-----------------------------:|:----:|:--------------------------------:|
| 1  |   10   |   30   |            0.003             | Adam | 2层全连接 |             [64]              | 8KB  |             0.98432              |
| 2  |   10   |   30   | <font color=Blue>0.03</font> | Adam | 2层全连接 |             [64]              | 8KB  |  <font color=Red>0.98693</font>  |
| 3  |   10   |   30   |            0.003             | Adam | 2层全连接 | <font color=Blue>[128]</font> | 13KB |  <font color=Red>0.99144</font>  |
| 4  |   10   |   30   |            0.003             | Adam | 2层全连接 | <font color=Blue>[32]</font>  | 6KB  | <font color=Green>0.98361</font> |
| 5  |   10   |   30   |            0.003             | SGD  | 2层全连接 | <font color=Blue>[32]</font>  | 6KB  | <font color=Green>0.98361</font> |

## 2 炼丹心得

一开始隐藏层维度是64，换成128分数降低了，说明过拟合数据了，之后换成32维，效果得到提升，说明对于这个数据集以及任务来说，32隐藏层大小的模型已经够用。

