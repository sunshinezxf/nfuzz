import seed.prioritizer.BatchPrioritizer as batchPrioritizer
import seed.prioritizer.BaseBatchPrioritizer as baseBatchPrioritizer
import muoperator.muop_util as mu_util
import muoperator.Mutator as mutator
from keras.datasets import mnist
import seed.BatchPool as batchPool
import seed.BaseSeedQueue as seedQueue
from keras.models import load_model
from keras import Model

# alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
ALPHA = 0.0
BETA = 0.2


# (x_train, y_train), (x_test, y_test) = mnist.load_data()

def load_seed():
    """
    加载种子数据集
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def preProcess(seeds):
    """
    对种子预处理
    :return:
    """

    return seeds

def evaluate(x_train, y_train, x_test, y_test, y_train_new, y_test_new):
    """
    评估变异结果
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param y_train_new:
    :param y_test_new:
    :return:
    """
    model = load_model('model/LeNet5.h5')

    # x_train, y_train, x_test, y_test, y_train_new, y_test_new = LeNet5.load_mnist()
    for layer in model.layers:
        for weight in layer.weights:
            print(weight.name, weight)

    layer_name = 'conv2d_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_test[0])
    print(intermediate_output)
    loss, accuracy = model.evaluate(x_test, y_test_new)
    print(loss, accuracy)


def main():
    """
    nfuzz的主流程
    :return:
    """

    # 加载种子
    x_train, y_train, x_test, y_test = load_seed()

    # 预处理
    seeds = preProcess(x_train)

    # 放入队列
    seed_queue = seedQueue(seeds)

    # batch选择器
    batch_prioritizer = baseBatchPrioritizer()

    # 存放待fuzz的种子
    batch_pool = batchPool(seed_queue=seed_queue, batch_size=32, p_min=0.1, gamma=1,
                           batch_prioritization=batch_prioritizer)

    batch_pool.pre_process()

    for i in range(10):
        # 随机选择一个batch进行变异
        batch=batch_pool.select_next()

        mu_batch=mu_util.batch_mutate(batch)

        # 变异后的种子加入pool
        batch_pool.add_batch(mu_batch)





if __name__ == '__main__':
    main()
