import numpy as np
import random
import myUtils.muop_util as mu_util
from coverage.NC import NeuronCoverages
from myUtils import model_utils
from seed.BaseSeedQueue import BaseSeedQueue
from seed.BatchPool import BatchPool
from seed.prioritizer.BaseBatchPrioritizer import BaseBatchPrioritizer


class DeepHunter:
    def __init__(self, model, seeds, max_loop=10):
        """
        :param model:待测模型
        :param seeds:带标签的种子列表
        :param max_loop: 循环次数
        """
        self.model = model
        self.seeds = seeds
        self.max_loop = max_loop
        self.coverage = 0
        self.neuron_coverage = NeuronCoverages(model)
        self.batch_pool = self.init_seeds()
        self.info = {}

    def init_seeds(self):
        """
        初始化种子的batch pool
        :return:
        """
        seed_queue = BaseSeedQueue(self.seeds)  # 放入队列
        batch_prioritizer = BaseBatchPrioritizer()  # batch选择器
        return BatchPool(seed_queue=seed_queue, batch_size=32, p_min=0.1, gamma=1,
                         batch_prioritization=batch_prioritizer)  # 存放待fuzz的种子

    def get_info(self, I):
        """
        一个种子I应该存储3个信息 todo:验证I是否能作为key
        I0:origin image
        I01:reference image,用了仿射变换的那个mutant
        state: 是否选过仿射变换
        :param I:
        :return:
        """
        if self.info.get(I) is None:
            self.info[I] = (I, I, 0)
        return self.info.get(I)

    @staticmethod
    def random_pick(a,b):
        """
        在所有算子中随机选择一种算子进行变异
        :return:算子序号
        """
        return random.randint(a, b)

    @staticmethod
    def is_satisfied(seed, mutant, alpha=0.02, beta=0.2):
        """
        判断mutant是否有意义
        :param alpha:
        :param beta:
        :param seed:
        :param mutant:
        :return:
        """
        # print('is satisfied??',seed.shape)
        height, width = seed.shape[:2]  # 获取图像的高和宽
        l0, l_inf = 0, 0
        for row in range(height):  # 遍历高
            for col in range(width):
                p_seed = seed[row, col]
                p_mutant = mutant[row, col]
                if p_seed != p_mutant:  # 像素值发生改变
                    l0 = l0 + 1

                minus = abs(int(p_seed) - int(p_mutant))
                if minus > l_inf:
                    l_inf = minus  # 像素值发生的最大改变

        if l0 < alpha * height * width:
            if l_inf <= 255:
                return True
            else:
                return False
        else:
            if l_inf < 255 * beta:
                return True
            else:
                return False

    def image_info(self, I):
        """

        :param I:
        :return:
        """
        return self.info[I]

    def mutate(self, I, try_num=3):
        """
        deepHunter alg2 图像变异
        :param try_num: 最大尝试次数
        :param I: 初始种子(单个图)
        :return: 变异成功的新种子或者原种子
        """
        (I0, I01, state) = self.get_info(I)

        for i in range(try_num):
            if state == 0:
                t = self.random_pick(0,6)
            else:
                t = self.random_pick(4,6)

            I1 = mu_util.transform(t, I)

            if self.is_satisfied(I01, I1):
                if t > 4:
                    state = 1
                    I01 = mu_util.transform(t, I0)
                self.info[I1] = (I0, I01, state)
                # 返回变异成功的种子
                return False, I1

        # 返回原种子
        return True, I

    def process(self):
        """
        deepHunter的主流程
        :return:
        """

        x_mutant = []
        y_mutant = []

        # 收集无效的变异种子
        failed_test = []

        for i in range(self.max_loop):
            print("epoch", i, '---------------------------')
            # 随机选择一个batch进行变异
            batch = self.batch_pool.select_next()

            # 筛选failedTest
            mu_batch = mu_util.batch_mutate(batch)
            valid_mu_batch, failed_mu_batch = model_utils.select_failed_test(mu_batch, self.model)

            # 收集无效的变异种子
            failed_test.append(failed_mu_batch)

            new_x_test = []
            new_y_test = []

            for j in range(len(valid_mu_batch)):
                new_x_test.append(valid_mu_batch[j][0])
                new_y_test.append(valid_mu_batch[j][1])
                x_mutant.append(valid_mu_batch[j][0])
                y_mutant.append(valid_mu_batch[j][1])
                # print(valid_mu_batch[j][0])
                # print(valid_mu_batch[j][1])

            # 计算神经元覆盖率 todo:batch包含多个输入，覆盖率计算如何进行
            basic_coverage = NeuronCoverages(self.model)
            for test_input in new_x_test:
                basic_coverage.update_coverage(np.array(test_input))
                coverage0 = basic_coverage.get_coverage()
                print("basic coverage:", coverage0)

            # 如果变异后的种子提升了覆盖率则加入pool
            if self.coverage_gain(new_x_test):
                self.batch_pool.add_batch(valid_mu_batch)

        print('mutation done')
        # 评估
        self.evaluate()
        # model_utils.evaluate(np.array(x_mutant), np.array(y_mutant), model)
        return x_mutant, y_mutant, failed_test

    def coverage_gain(self, batch) -> bool:
        """
        判断变异后的种子是否提高了覆盖率
        :param batch:
        :return:
        """
        self.neuron_coverage.update_batch_coverage(batch)
        cov = self.neuron_coverage.get_coverage()
        if self.coverage < cov['batch_neuron_coverage']:
            self.coverage = cov['batch_neuron_coverage']  # 更新覆盖率
            return True
        return False

    def evaluate(self):
        pass
