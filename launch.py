from keras.datasets import mnist
from keras.models import load_model
from engine import deep_hunter, entropic
from seed import BaseSeedQueue as sq
from seed import BatchPool as bp
from seed.prioritizer import BaseBatchPrioritizer as bbp


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

    # 放入队列
    seed_queue = sq.BaseSeedQueue(seeds)

    # batch选择器
    batch_prioritizer = bbp.BaseBatchPrioritizer()

    # 存放待fuzz的种子
    batch_pool = bp.BatchPool(seed_queue=seed_queue, batch_size=32, p_min=0.1, gamma=1,
                              batch_prioritization=batch_prioritizer)

    deep_hunter.process(batch_pool, model)
    entropic.process(seeds,model)


if __name__ == '__main__':
    main()
