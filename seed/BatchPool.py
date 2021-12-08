import os
import random
import uuid
import cv2
import numpy as np
from pathlib import Path
from myUtils.converter_utils import path_2_ndarray_convert
from seed.prioritizer.BaseBatchPrioritizer import BaseBatchPrioritizer
from seed import Seed
from seed import PriorityPool

'''
用来存放待fuzz的种子
'''


class BatchPool:
    # SeedQueue
    seed_queue = None

    # Batch Size
    batch_size = None

    # BatchPrioritization
    batch_prioritization = None

    # Batch Buffer
    # If size of batch buffer equals the batch_size, add batch into pool
    batch_buffer = []

    # The minimum probability,
    p_min = None

    # γ is the weight, influence the probability of P(B)
    # Here, gamma represents the number of batches with the most fuzzed times
    gamma = None

    # Pool
    # element type is '{fuzzedTimes, batch}'
    pool = PriorityPool.PriorityPool()

    # constructor
    def __init__(self, seed_queue, batch_size=32, p_min=0.1, gamma=1, batch_prioritization=BaseBatchPrioritizer()):
        self.batch_prioritization = batch_prioritization
        self.gamma = gamma
        self.p_min = p_min
        self.batch_size = batch_size
        self.seed_queue = seed_queue
        self.pre_process()

    def pre_process(self):
        """
            把种子封装以后分批放入batch_buffer，存到pool
        """
        while not self.seed_queue.empty():
            self.batch_buffer.append(self.seed_queue.pop())
            if len(self.batch_buffer) == self.batch_size:
                element = {
                    "fuzzed_times": 0,
                    # "batch": np.array(self.batch_buffer)
                    "batch": self.batch_buffer
                }

                # 把整个batch封装到seed中
                seed=Seed.Seed(element,self.probability(element["fuzzed_times"]))
                self.pool.push(seed)
                self.batch_buffer = []

    def select_next(self):
        """
            按照优先级从优先队列pool中选取一个batch
            :return
                seed.val() 即element
                batch -- a batch of seeds
            :except
                StopIteration -- The queue is empty
        """
        if self.pool.empty():
            raise StopIteration("The pool is empty.")

        # return self.pool.pop()
        return self.pool.pop()["batch"]

        # while True:
        #     element = random.choice(self.pool)
        #     probability = self.batch_prioritization.probability(element["fuzzed_times"], self.p_min, self.gamma)
        #
        #     if probability >= random.random():  #
        #         element["fuzzed_times"] = element["fuzzed_times"] + 1
        #         self.gamma = max(element["fuzzed_times"], self.gamma)
        #         return element["batch"]

    # def get_pool(self):
    #     """
    #     返回pool中的所有种子
    #     :return: 一维的种子列表
    #     """
    #
    #     ret = []
    #
    #     for batch in self.pool:
    #         for seed in batch:
    #             ret.append(seed)
    #
    #     return ret

    def save(self, path, start_with=0, save_size=-1):
        """
            Save images to local
            :param
                path -- directory path
                start_with -- start index of pool
                save_size -- the number of batches to save, -1 represent save all batches
            :return
                number -- number of batch saved success
            :except
                IOError -- File not exists | File not a directory
                ValueError -- The save_size must be greater than or equal to -1
                IndexError -- The start_with must be greater than or equal to 0
        """
        file_path = Path(path)
        if not file_path.exists():
            raise IOError("File not exists, path=" + path)
        if not file_path.is_dir():
            raise IOError("File not a directory, path=" + path)
        if save_size < -1:
            raise ValueError("The save_size must be greater than or equal to -1. save_size=" + str(save_size))
        if start_with < 0:
            raise IndexError("The start_with must be greater than or equal to 0. start_with=" + str(start_with))

        if save_size == -1:
            end = self.pool.size()
        else:
            end = min(self.pool.size(), start_with + save_size)

        for i in range(start_with, end):
            # batch = self.pool[i]["batch"]
            batch=self.pool.pop()["batch"]
            for img in batch:
                uuid4 = str(uuid.uuid4()) + ".png"
                suid = ''.join(uuid4.split('-'))
                path0 = os.path.join(path, suid)
                cv2.imwrite(path0, img)
        return end - start_with

    def add_seed_by_path(self, path_list):
        """
            add seeds into seed queue
            根据路径
            :param
                path_list -- seed path list
            :return
                num -- number of seeds saved success
        """
        num = 0
        for path in path_list:
            self.seed_queue.push(path_2_ndarray_convert(path))
            num = num + 1
        return num

    def add_seed(self, seed):
        """
        添加单个种子
        :param seed:
        :return:
        """
        self.seed_queue.push(seed)

    def add_batch(self, batch):
        """
        把变异后的种子加回来
        :param batch:
        :return:
        """
        element = {
            "fuzzed_times": 0,
            # "batch": np.array(batch)
            "batch": batch
        }
        # self.pool.append(element)
        self.pool.push(Seed.Seed(element,self.probability()))

    def random_generate(self, number, generator):
        """
            random generate seeds and add into seed queue
            :param
                number -- the number of randomly generated seeds
                generator -- randomly seed generator
            :return
                seeds -- list of randomly generated seeds
        """
        seed_list = []
        for i in range(number):
            seed = generator.generate()
            self.seed_queue.push(seed)
            seed_list.append(seed)
        return seed_list

    def probability(self,fuzzed_time=0):
        """
        优先级选取规则
        :param fuzzed_time: fuzz次数
        :return:
        """
        return self.batch_prioritization.probability(fuzzed_time,self.p_min,self.gamma)