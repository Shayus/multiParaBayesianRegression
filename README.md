# Bayesian Optimization for multi Parameters

## Start

针对自动图像生成和美学评价网络，设计了一个基于贝叶斯优化的回归模型

## Usage

输入参数较多， 一维的优化方法不太适用，选择高斯过程来模拟原函数

input：

for train

``./main.py patternType style tilingType cutStyle Q scale Value``

output:

the next position's value from AI 

输出类似

    1

    2
    
    3

    4

    5

    6

当输出变更为

    The model has been trained

此时训练已经完成，不再依赖神经网络

for test  

``./main.py patternType style tilingType cutStyle Q scale``

输出类似(贝叶斯预估图像美学分)

    5