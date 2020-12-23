先参考deephunter,它的目的是对模型进行模糊测试,主要实现的是扰动方法和评估部分.
扰动方法我们需要先实现一些经典的扰动方法,然后后续提出自己的.评估部分也是如此，
会先有一些基于神经元激活情况的覆盖准则的实现,然后把我们之前PPT里面提到的基于熵值的评估方法加进去.
神经元覆盖 Neuron Coverage (NC):阈值自己定，例如0、0.25、0.75
k-多节神经元覆盖 k-Multisection Neuron Coverage (KMNC)
神经元边界覆盖 Neuron Boundary Coverage (NBC)
强神经元激活覆盖 Strong Neuron Activation Coverage (SNAC)
Top-k神经元覆盖 Top-k Neuron Coverage (TKNC)

![image-20200814172936466](C:\Users\MI\AppData\Roaming\Typora\typora-user-images\image-20200814172936466.png)