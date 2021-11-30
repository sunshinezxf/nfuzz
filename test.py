import myUtils.muop_util as mu
from keras.datasets import mnist
import numpy as np

# 初始种子
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print(y_train[0])
seeds=(x_train[0],y_train[0])

print(seeds[0])

# # 重新定义数据格式，归一化
# x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
# x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#
# # # 转one-hot标签
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)

# 对MNIST数据做简单修改，添加噪音
def do_basic_mutations(element, a_min=-1.0, a_max=1.0):

    image, label = element
    sigma = 0.5
    noise = np.random.normal(size=image.shape, scale=sigma)

    mutated_image = noise + image

    mutated_image = np.clip(
        mutated_image, a_min=a_min, a_max=a_max
    )


    mutated_element = [mutated_image, label]
    return mutated_element

# 进行变异
def mutate(element):
    image, label = element

    mutated_image=mu.image_mutate(5,image)

    mutated_element = [mutated_image, label]
    return mutated_element

