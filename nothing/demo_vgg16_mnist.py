from keras.datasets import mnist
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.applications import VGG16, inception_v3, resnet50, mobilenet
import keras
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

(x_train,y_train),(x_test,y_test)=mnist.load_data()

# 建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层（全连接层）去掉，只保留其余的网络
# 结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
# VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
# VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器
model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))#输入进来的数据是48*48 3通道
#选择imagnet,会选择当年大赛的初始参数
#include_top=False 去掉最后3层的全连接层看源码可知
for layer in model_vgg.layers:
    layer.trainable = False#别去调整之前的卷积层的参数
model = Flatten(name='flatten')(model_vgg.output)#去掉全连接层，前面都是卷积层
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax')(model)#model就是最后的y
model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
#把model_vgg.input  X传进来
#把model Y传进来 就可以训练模型了

# 打印模型结构，包括所需要的参数
model_vgg_mnist.summary()

x_train, y_train = x_train[:1000], y_train[:1000]#训练集1000条
x_test, y_test = x_test[:100], y_test[:100]#测试集100条

# 转成vgg16可用的图像
def vgg16_convert(imgs):
    imgs = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in imgs]  # 变成RGB的
    imgs = np.concatenate([arr[np.newaxis] for arr in imgs]).astype('float32')
    return imgs


X_train = vgg16_convert(x_train)
X_test = vgg16_convert(x_test)

print(X_train.shape)
print(X_test.shape)

X_train = X_train / 255
X_test = X_test / 255

#one-hot热点化
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

sgd = SGD(lr=0.05, decay=1e-5)#lr 学习率 decay 梯度的逐渐减小 每迭代一次梯度就下降 0.05*（1-（10的-5））这样来变
#随着越来越下降 学习率越来越小 步子越小
model_vgg_mnist.compile(loss='categorical_crossentropy',
                                 optimizer=sgd, metrics=['accuracy'])

model_vgg_mnist.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe),
                             epochs=50, batch_size=50)


scores = model_vgg_mnist.evaluate(X_test, y_test_ohe, verbose=0)
print("Accuracy: %.2f%%"% (scores[1]*100))

# 保存模型
model_vgg_mnist.save('demo_model.h5')