from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils import np_utils
import numpy as np
import cv2

# vgg_model = vgg16.VGG16(weights='imagenet')
# inception_model = inception_v3.InceptionV3(weights='imagenet')
# resnet_model = resnet50.ResNet50(weights='imagenet')
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')

#加载数据集
from tensorflow.python.keras.optimizers import SGD


def load_mnist():
    path = r'mnist.npz'  # 放置mnist.py的目录。注意斜杠
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)



(X_train_data, Y_train), (X_test_data, Y_test) = load_mnist()  # 下载数据
X_train_data = X_train_data.astype('float32')  # uint8-->float32
X_test_data = X_test_data.astype('float32')
X_train_data /= 255  # 归一化到0~1区间
X_test_data /= 255
# (60000, 48, 48, 3)
X_train = []
# (10000, 48, 48, 3)
X_test = []
# 把(27, 27, 1)维的数据转化成(48, 48, 3)维的数据
for i in range(X_train_data.shape[0]):
    X_train.append(cv2.cvtColor(cv2.resize(X_train_data[i], (48, 48)), cv2.COLOR_GRAY2RGB))
for i in range(X_test_data.shape[0]):
    X_test.append(cv2.cvtColor(cv2.resize(X_test_data[i], (48, 48)), cv2.COLOR_GRAY2RGB))

X_train = np.array(X_train)
X_test = np.array(X_test)
# 独热编码
y_train = np_utils.to_categorical(Y_train, num_classes=10)
y_test = np_utils.to_categorical(Y_test, num_classes=10)

# 构建网络
vgg16_model =vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

for layer in vgg16_model.layers:
    layer.trainable = False # 别去调整之前的卷积层的参数

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.4))
top_model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)
# sgd = SGD(learning_rate=0.05, decay=1e-5)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=15)
score=model.evaluate(X_test, y_test)
print(score.history)
X_test=X_test[:10]
res=model.predict(X_test,batch_size=2,verbose=1)
print(res)














