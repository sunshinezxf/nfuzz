from seed.prioritizer import BaseBatchPrioritizer as bbp
import myUtils.muop_util as mu_util
from keras.datasets import mnist
from seed import BatchPool as bp
from seed import BaseSeedQueue as sq
from keras.models import load_model
from myUtils import model_utils
from engine import deep_hunter,entropic

# alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
ALPHA = 0.0
BETA = 0.2

global_model = load_model('./model/LeNet5.h5')


# (x_train, y_train), (x_test, y_test) = mnist.load_data()

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

    deep_hunter.process(batch_pool, global_model)
    entropic.process(seeds,global_model)


if __name__ == '__main__':
    main()
