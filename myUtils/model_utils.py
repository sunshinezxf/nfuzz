from keras.utils import np_utils
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras import Model
import numpy as np
import cv2

def select_failed_test(mutate_seeds, model):
    """
    筛选failedTest
    :param mutate_seeds: 变异后的种子集
    :param model: 待测的keras模型
    :return:
    """
    valid_mu_batch = []
    failed_mu_batch = []

    for seed in mutate_seeds:
        if is_failed_test(seed, model):
            valid_mu_batch.append(seed)
        else:
            failed_mu_batch.append(seed)

    return valid_mu_batch, failed_mu_batch


def is_failed_test(seed, model):
    """
    判断是否有效变异
    :param seed:
    :param model:
    :return:
    """
    test_img = seed[0]
    test_label = seed[1]
    # 预测标签
    # 获取第一层输入的shape
    first_layer = model.get_layer(index=0)
    input_shape = first_layer.input_shape
    # 一般来说输入的shape是三维的
    if len(input_shape) == 4:
        test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1], 1)
    elif len(input_shape) == 3:
        test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], 1)
    predict_label = model.predict_classes(test_img)[0]

    if test_label == predict_label:
        return True
    else:
        return False


def input_reshape(x_test, model):
    """
    根据具体模型对输入进行reshape
    :param x_test:
    :param model:
    :return:
    """
    # 获取第一层输入的shape
    first_layer = model.get_layer(index=0)
    input_shape = first_layer.input_shape
    # 一般来说输入的shape是三维的
    if len(input_shape) == 4:
        print(x_test.shape)
        x_test = x_test.reshape(-1, x_test.shape[0], x_test.shape[1], x_test.shape[2])
    return x_test


def print_model_config(model):
    """
    打印模型信息
    :param model:
    :return:
    """
    for layer in model.layers:
        print(layer.name, ':input_shape=', layer.input_shape)
        # print('weights:-------------------------------')
        # for weight in layer.weights:
        #     print(weight.name, weight)


def evaluate(x_test, y_test, model):
    """
    通用的模型评估。暂时基于LeNet5
    :param x_test:
    :param y_test:
    :param model:
    :return:
    """
    print_model_config(model)

    x_test = input_reshape(x_test, model)
    x_test /= 255
    # 归一化
    y_test_new = np_utils.to_categorical(num_classes=10, y=y_test)
    loss, accuracy = model.evaluate(x_test, y_test_new)
    print('model evaluate:-------------------------------')
    print('loss', loss)
    print('accuracy', accuracy)


def vgg_evaluate(x_train, y_train, x_test, y_test):
    """
    建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层（全连接层）去掉，只保留其余的网络
    结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
    VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
    VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """

    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))  # 输入进来的数据是48*48 3通道
    # 选择imagnet,会选择当年大赛的初始参数
    # include_top=False 去掉最后3层的全连接层看源码可知
    for layer in model_vgg.layers:
        layer.trainable = False  # 别去调整之前的卷积层的参数
    model = Flatten(name='flatten')(model_vgg.output)  # 去掉全连接层，前面都是卷积层
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax')(model)  # model就是最后的y
    model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
    # 把model_vgg.input  X传进来
    # 把model Y传进来 就可以训练模型了

    # 打印模型结构，包括所需要的参数
    model_vgg_mnist.summary()

    x_train, y_train = x_train[:1000], y_train[:1000]  # 训练集1000条
    x_test, y_test = x_test[:100], y_test[:100]  # 测试集100条

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

    # one-hot热点化
    def tran_y(y):
        y_ohe = np.zeros(10)
        y_ohe[y] = 1
        return y_ohe

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

    sgd = SGD(lr=0.05, decay=1e-5)  # lr 学习率 decay 梯度的逐渐减小 每迭代一次梯度就下降 0.05*（1-（10的-5））这样来变
    # 随着越来越下降 学习率越来越小 步子越小
    model_vgg_mnist.compile(loss='categorical_crossentropy',
                            optimizer=sgd, metrics=['accuracy'])

    model_vgg_mnist.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe),
                        epochs=50, batch_size=50)

    scores = model_vgg_mnist.evaluate(X_test, y_test_ohe, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # 保存模型
    model_vgg_mnist.save('../model/demo_model.h5')


def Lenet5_evaluate(x_train, y_train, x_test, y_test, model):
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    y_train = y_train.astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_test = x_test.astype("float32")
    y_test = y_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train_new = np_utils.to_categorical(num_classes=10, y=y_train)
    y_test_new = np_utils.to_categorical(num_classes=10, y=y_test)

    history = model.fit(x_train, y_train_new, batch_size=64, epochs=2, verbose=1, validation_split=0.2, shuffle=True)
    loss, accuracy = model.evaluate(x_test, y_test_new)
    print(loss, accuracy)
    print(history.history)

    # -> https://github.com/keras-team/keras/issues/2378
    # model.save("model/LeNet5.h5", overwrite=True)