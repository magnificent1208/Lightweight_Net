# attention transfer

## 1 注意力机制

1.1 目的： 有效提高CV领域的模型性能。
1.2 在网络轻量化的应用：结合知识蒸馏。
1.3 回顾现有的网络轻量化方法：
1.3.1 人工设计: SqueezeNet/MobileNet/ShuffleNet/Xception
1.3.2 NAS: 网络自动搜索最优化配置
1.3.3 模型压缩：知识蒸馏/剪枝(减去不需要的CNN通道)/量化(fp16-int8)/低秩分解 
1.4 注意力迁移 Attention Transfer 方法。将教师网络学习得到的注意力图(ATTENTION MAP)作为知识信息，蒸馏到学生网络，并其逐渐模拟。
## 2 基于激活的注意力转移 Activation-based
在前馈过程中，提取attention map作为学习目标。
通道维度压缩，把3d fm 拍成一张2d图，这张图就是attention map
- sum of abs value 把每个通道的绝对值加起来
- sum of abs raised to the power of p 把通道绝对值的p次方加起来
- max of abs raised to the power of p 最大值的p次方加起来
**p又是一个超参 emmmm &考虑到计算量，手动设计压缩方式，而不是如同SENet
## 3 基于梯度的注意力转移
在反向传播的过程中，提取梯度图作为学习目标。
（同2 只不过换成了梯度图）
