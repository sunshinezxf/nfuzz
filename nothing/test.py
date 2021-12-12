from keras.models import load_model
from keras.datasets import mnist


def load_seed():
    """
    加载种子数据集
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def pre_process(x_train, y_train):
    """
    对种子进行预处理预处理
    :return:
    """
    seeds = []
    for i in range(len(x_train)):
        seed = (x_train[0], y_train[0])
        seeds.append(seed)
    return seeds


# 加载种子
x_train, y_train, x_test, y_test = load_seed()

# 预处理
seeds = pre_process(x_train, y_train)

dict={}
print(type(seeds[0]))
dict[seeds[0]]='1'
dict[seeds[1]]='2'
print(dict.values())