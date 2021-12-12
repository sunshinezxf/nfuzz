import os
import random
import uuid
import copy
import cv2
import numpy as np
from pathlib import Path
from myUtils.converter_utils import path_2_ndarray_convert
from seed.prioritizer.BaseBatchPrioritizer import BaseBatchPrioritizer
from seed.Seed import Seed
from seed.Batch import Batch

class BatchPool:
    def __init__(self, seed_queue, batch_size=32, p_min=0.1, gamma=1, batch_prioritization=BaseBatchPrioritizer()):
        """
        :param seed_queue:
        :param batch_size: If size of batch buffer equals the batch_size, add batch into pool
        :param p_min: The minimum probability
        :param gamma: γ is the weight, influence the probability of P(B)
        Here, gamma represents the number of batches with the most fuzzed times
        :param batch_prioritization:
        """
        self.batch_prioritization = batch_prioritization
        self.gamma = gamma
        self.p_min = p_min
        self.batch_size = batch_size
        self.seed_queue = seed_queue
        self.pool = []
        self.pre_process()

    def pre_process(self):
        """
        把种子封装以后分批放入batch_buffer，存到pool
        """
        batch_buffer = []
        while not self.seed_queue.empty():
            batch_buffer.append(self.seed_queue.pop())
            if len(batch_buffer) == self.batch_size:
                # 把整个batch封装到seed中
                batch = Batch(copy.deepcopy(batch_buffer), 0)
                self.pool.append(batch)
                batch_buffer = []

        # 剩余不够size的也打包进pool
        if len(batch_buffer) > 0:
            batch = Batch(copy.deepcopy(batch_buffer), 0)
            self.pool.append(batch)

    def select_next(self) -> Batch:
        """
        按照优先级从优先队列pool中选取一个batch
        :return：batch对象
        :except：StopIteration -- The queue is empty
        """
        if len(self.pool) == 0:
            raise StopIteration("The pool is empty.")
        prioritization=[]
        for batch in self.pool:
            prioritization.append(batch.probability())

        chosen = random.choices(self.pool,weights=prioritization,k=1)[0]
        # 更新fuzzed_times和gamma
        chosen.fuzzed_times += 1
        self.gamma=max(chosen.fuzzed_times,self.gamma)
        return chosen

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
            end = len(self.pool)
        else:
            end = min(len(self.pool), start_with + save_size)

        for i in range(start_with, end):
            # batch = self.pool[i]["batch"]
            batch = self.pool.pop()
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
        self.pool.append(batch)

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

    def probability(self, fuzzed_time=0):
        """
        优先级选取规则
        :param fuzzed_time: fuzz次数
        :return:
        """
        return self.batch_prioritization.probability(fuzzed_time, self.p_min, self.gamma)
