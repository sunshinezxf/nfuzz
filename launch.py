from keras.datasets import mnist
from keras.models import load_model
from engine.deep_hunter import DeepHunter
from engine.entropic import Entropic


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


def main():
    """
    主流程
    todo：entropic和deepHunter如何进行整合
    :return:
    """

    # 加载模型
    model = load_model('./model/LeNet5.h5')

    # 加载种子
    x_train, y_train, x_test, y_test = load_seed()

    # 预处理
    seeds = pre_process(x_train, y_train)

    deep_hunter = DeepHunter(model,seeds)
    deep_hunter.process()
    # entropic.process(seeds,model)


if __name__ == '__main__':
    main()
