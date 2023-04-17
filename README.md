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

# Attention
由于系数较多， 且系数变化范围大， 本方法前期运算速度很慢

代码采用int， 不深入浮点数的情况下，输入一组参数需要运算超过50G次运算

建议缩小范围


参数 参考

	patterntype    //0 or 1 int

	Style   //0 - 111 int

	tilingtype   //0 - 211 int

	Cutstyle    //0 - 42 int                                “1”印花（0-18）； “2”提花 （19-32） ；“3” 针织（33-42）  
	
	Q    //0 - 10 float  

	Scale    //12 - 18 int

	Xmin    //-10 - +10 float

	Ymin   //-10 - +10 float