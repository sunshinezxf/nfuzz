import numpy as np
from muoperator.Mutator import Mutator
import cv2
import random


class ScaleMutator(Mutator):
    """
        A mutator for image scale
    """
    def __init__(self):
        super().__init__()

    def mutate(self, seed):
        """
            将图像沿着横纵轴缩放
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        # print(seed.shape)
        height, width = seed.shape[:2]  # 获取图像的高和宽
        x_rate = random.random() * 5
        y_rate = random.random() * 5
        M = np.array([
            [x_rate, 0, 0],
            [0, y_rate, 0]
        ], dtype=np.float32)
        mutant = cv2.warpAffine(seed, M, (height, width))
        return mutant
