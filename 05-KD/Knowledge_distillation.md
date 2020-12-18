

# 知识蒸馏

主要介绍源于2015年Hinton发表的神作：Distilling the Knowledge in a Neural Network

## 1 Ensemble Model 集成模型

集成学习是一种机器学习的范式，通过结合来自多个模型的决策，以提高最终的稳定性和准确性。汇聚结果的方法最简单是投票(majority vote)或者加权。
*Ensemble learning* refers to a method where many base models are combined to carry out the same task. 

### 代表性工作：
#### a. bagging *random sampling with replacement
​	stage1 bootstrap sampling
​	stage 2 model training
​	stage 3 model forecasting
​	stage 4 result aggregating 
#### b. boosting *random sampling with replacement over weighted data
​	使用全部样本，每轮改变样本的权重。
​	次轮的目标是找到一个函数拟合上一轮的残差。
​	Boosting会较小在上一轮的训练正确的样本的权重，增大错误样本的权重。（weighted data)
### 缺点：
​	模型可解释性差。
​	算力消耗大，运算时间长，不符合移动端需求。
​	模型选择有随机性，不能确保是最佳组合。

## 2 知识蒸馏思想
### 动机
移动端部署，孵化一个体量小，性能高的模型。
teacher-student模式。从集成学习当中得到启发。
训练的多个网络，都是统一的教师网络，最后输出一个学生网络。期望在拥有更少参数，更小规模的情况下，达到与教师网络相似，甚至超越的精度要求。

### 知识的概念
Knowledge -- 模型参数信息。如何输入向量映射到输出向量。
Example:
Tag [cow,dog,cat,car]
在结果输出Prediction[0.05,0.3,0.2,0.005]输出概率当中，其实也隐含了当前对象 0.3像狗，0.2像猫的”知识“。#详细请看后续关于“hard target &soft targets 的讨论”

## 3 知识蒸馏方法
### 3.1 logits概念
概念：全连接层->logits-> softmax *即FC的输出，softmax的输入，最后得到概率0-1。
### 3.2 方法
​	S1 教师网络输出概率分布->
​	S2 ->学生网络(使用损失函数)模拟教师网络输出概率分布
​	S3 ->设计学生网络的LOSS最小化以上两个概率分布的差距。
### 3.3 温度参数T（temperature）
本质就是在softmax中加入一个超参，使得输出类别概率分布越“soft”。
$$
\mathrm{q}_{\mathrm{i}}=\frac{\exp \left(z_{i} / T\right)}{\sum_{\mathrm{j}} \exp \left(z_{j} / T\right)}
$$
​	T的效果：
​		T = 1，公式同softmax输出后的结果；
​		T越接近0，公式越同One-hot编码，最大值趋向于1，最小值趋向于0；
​		提升T，则保留更多类别关系信息。
*** 讨论#1 关于hard target 和soft target的讨论
​	Hard Target: one-hot; 0,0,1,0,0; 信息量小
​	Soft Target softmax后概率：0,0.05,0.3,0.1,0; 信息量大
​		软目标的优势：
​			弥补了简单分类中监督信号不足（信息熵比较少）的问题，增加了信息量；
​			提供了训练数据中类别之间的关系(数据增强)；
​			可能增强了模型泛化能力。**这个略扯淡感觉
*** 讨论#2 关于label smoothing regularization(LSR)的讨论
​	目的是对hard target的优化。
​	即标签平滑正则化：在输出Y中添加噪声，实现对模型约束，降低模型过拟合。
​	https://www.jianshu.com/p/6a5ea4ddbf32
经过升温之后效果：
真实标签                                [0, 1, 0, 0]
原始softmax输出概率            [10-6, 0.9, 0.1, 10-9]
升温后softmax输出概率         [0.05, 0.3, 0.2, 0.005]
*使得soft target变得平坦，号称可以从非正确预测中提取有用信息，供学生网络学习。

### 3.4 蒸馏过程
3.4.1 几个基础概念梳理
- Input 输入数据

- Teacher model 教师网络模型

- student model

- soft target 教师网络升温后的softmax预测类别输出

- soft prediction 学生网络升温后的softmax预测类别输出

- hard prediction 温度系数为1，即学生网络原始softmaxyuce shuchu 

- hard target 输入数据真实标签

  ![image-20201215162601590](C:\Users\maggie\AppData\Roaming\Typora\typora-user-images\image-20201215162601590.png)
  3.4.2 具体过程
  S1 教师网络训练。利用数据训练一个层数深，提取能力强的网络，得到logits后，利用升温T的softmax得到预测类别概率分布softtargets
  S2 蒸馏教师网络知识到学生网络。构造distillation loss和student loss,加权相加作为最终loss

  - distillation loss 学生网络尽可能拟合教师网络。
  - student loss 使用真实标签(GT)矫正教师网络中的错误。
  - 公式推导：

  $$
  \boldsymbol{L}=\alpha \boldsymbol{L}_{\text {soft}}+\boldsymbol{\beta} \boldsymbol{L}_{\boldsymbol{h} \boldsymbol{a r d}}
  $$
  $$
  L=\gamma C E(y, p)+(1-\gamma) T^{2} C E(q, p)
  $$

  -- dist loss：交叉熵/KL散度

  -- stud loss: 交叉熵

  -- y : 真实标签

  -- p ：学生网络预测概率

  -- q : 教师网络预测概率

  -- gamma & （1-gamma）: 人为设置的两个损失函数权重

  -- T：温度参数

  ​	公式推导参考链接：

  ​	关于损失函数的基础 https://zhuanlan.zhihu.com/p/35709485


## 4 训练专家集成模型
### 4.1 目的
通过知识蒸馏，训练包含专家模型的集成模型，减少计算量。
### 4.2 结构
集成模型包含一个通用模型(generlist model)和多个专家模型(sepcialist model).
- generlist model
	负责将数据进行初略的区分，将相似的图片归为一类。
- sepcialist model
	负责将数据进行细致的分类。
	训练专家模型的时候，模型的参数初始化时完全复制通用模型的值，这样可以保留通用模型的而所有知识。
## 5 实验设置以及结果分析

