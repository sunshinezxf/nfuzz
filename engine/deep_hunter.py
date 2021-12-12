import numpy as np
import random
import math
from coverage.NC import NeuronCoverages
from myUtils import model_utils
from seed.BaseSeedQueue import BaseSeedQueue
from seed.BatchPool import BatchPool
from seed.Batch import Batch
from seed.prioritizer.BaseBatchPrioritizer import BaseBatchPrioritizer
from muoperator.scale_mutator import ScaleMutator
from muoperator.translation_mutator import TranslationMutator
from muoperator.shear_mutator import ShearMutator
from muoperator.transpose_mutator import TransposeMutator
from muoperator.noise_mutator import NoiseMutator
from muoperator.blur_mutator import BlurMutator
from muoperator.contrast_mutator import ContrastMutator


class DeepHunter:
    def __init__(self, model, seeds, max_loop=10):
        """
        :param model:待测模型
        :param seeds:种子tuple列表,(image,label),image为nd_array
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
        一个种子I应该存储3个信息 todo:nd_array类型的I不能能作为key
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
        :param alpha:>0
        :param beta:<1
        :param seed:
        :param mutant:
        :return:
        """
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

    @staticmethod
    def mutation_potential(I,I01,beta=0.2):
        """
        fuzzer的调度指标
         the mutation potential approximately represents the mutation space of an image
        :param I:original image
        :param I01:mutated image
        :param beta:
        :return:
        """
        height, width = I.shape[:2]
        diff_sum=0
        for row in range(height):
            for col in range(width):
                pixel_I = I[row, col]
                pixel_I01 = I01[row, col]
                diff_sum += abs(pixel_I-pixel_I01)
        return beta*255*height*width-diff_sum

    @staticmethod
    def power_schedule(seeds, k) -> dict:
        """
        把k次mutation的次数分配给seeds
        potential高的seeds分配的次数多
        :param seeds:
        :param k:
        :return:
        """
        power_sum=0
        schedule = {}
        for seed in seeds:
            power_sum += seed.potential

        for seed in seeds:
            schedule[seed]=math.floor((seed.potential/power_sum)*k)

        return schedule




    @staticmethod
    def transform(state, seed):
        """
        所有的变异算子
        :param state:
        :param seed:
        :return: 变异后的图片
        """
        if state == 0:    # 缩放
            return ScaleMutator().mutate(seed)
        elif state == 1:  # 平移
            return TranslationMutator().mutate(seed)
        elif state == 2:  # 剪裁
            return ShearMutator().mutate(seed)
        elif state == 3:  # 旋转
            return TransposeMutator().mutate(seed)
        elif state == 4:  # 噪声
            return NoiseMutator().mutate(seed)
        elif state == 5:  # 模糊
            return BlurMutator().mutate(seed)
        else:             # 对比度/亮度
            return ContrastMutator().mutate(seed)

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
        :return: (is_changed,变异成功的新种子或者原种子)
        """
        (I0, I01, state) = self.get_info(I)

        for i in range(try_num):
            if state == 0:
                t = self.random_pick(0,6)
            else:
                t = self.random_pick(4,6)

            I1 = self.transform(t, I)

            if self.is_satisfied(I01, I1):
                if t > 4:
                    state = 1
                    I01 = self.transform(t, I0)
                self.info[I1] = (I0, I01, state)
                # 返回变异成功的种子
                return True, I1

        # 返回原种子
        return False, I

    def batch_mutate(self,batch):
        """
        返回变异完的一批种子,failedTest的判断逻辑由外部实现
        :param batch:
        :return:
        """
        valid_test = []
        result_set = []

        for img_tuple in batch:
            flag, img, label = self.mutate(img_tuple[0], img_tuple[1])
            if flag:
                # 是原种子，无需验证
                valid_test.append((img, label))
            else:
                result_set.append((img, label))
        return result_set

    def process(self,batch_size,K):
        """
        deepHunter的主流程
        :param batch_size: 每个batch进行mutate的种子数
        :param K:一个batch的mutation总次数
        :return:
        """

        x_mutant = []
        y_mutant = []

        # 收集无效的变异种子
        failed_test = []

        for i in range(self.max_loop):
            print("epoch", i, '---------------------------')
            # 选择一个batch进行变异
            batch = self.batch_pool.select_next()

            sampled_seeds=random.sample(batch.val(),batch_size)

            Ps=self.power_schedule(sampled_seeds,K)

            new_batch=[]

            for I in sampled_seeds:
                for n in range(Ps[I]):
                    is_changed,mutant=self.mutate(I)
                    if model_utils.is_failed_test(mutant,self.model):
                        failed_test.append(mutant)
                    elif is_changed:
                        new_batch.append(mutant)

            if self.coverage_gain(new_batch):  # todo:predict or coverage ?
                self.batch_pool.add_batch(Batch(new_batch,0))

        #     # 筛选failedTest
        #     mu_batch = self.batch_mutate(batch)
        #     valid_mu_batch, failed_mu_batch = model_utils.select_failed_test(mu_batch, self.model)
        #
        #     # 收集无效的变异种子
        #     failed_test.append(failed_mu_batch)
        #
        #     new_x_test = []
        #     new_y_test = []
        #
        #     for j in range(len(valid_mu_batch)):
        #         new_x_test.append(valid_mu_batch[j][0])
        #         new_y_test.append(valid_mu_batch[j][1])
        #         x_mutant.append(valid_mu_batch[j][0])
        #         y_mutant.append(valid_mu_batch[j][1])
        #         # print(valid_mu_batch[j][0])
        #         # print(valid_mu_batch[j][1])
        #
        #     # 计算神经元覆盖率
        #     basic_coverage = NeuronCoverages(self.model)
        #     for test_input in new_x_test:
        #         basic_coverage.update_coverage(np.array(test_input))
        #         coverage0 = basic_coverage.get_coverage()
        #         print("basic coverage:", coverage0)
        #
        #     # 如果变异后的种子提升了覆盖率则加入pool
        #     if self.coverage_gain(new_x_test):
        #         self.batch_pool.add_batch(valid_mu_batch)
        #
        # print('mutation done')
        # # 评估
        # self.evaluate()
        # # model_utils.evaluate(np.array(x_mutant), np.array(y_mutant), model)
        # return x_mutant, y_mutant, failed_test

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
