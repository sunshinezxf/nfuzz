import numpy as np
from muoperator.Mutator import Mutator
import cv2
import random


class TranslationMutator(Mutator):
    """
        A mutator for image translation
    """
    def mutate(self, seed):
        """
            平移图像， x_dis,y_dis可为负数
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        height, width = seed.shape[:2]  # 获取图像的高和宽
        x_dis = random.randint(0, width)-(width/2)
        y_dis = random.randint(0, height)-(height/2)
        M = np.array([
            [1, 0, x_dis],
            [0, 1, y_dis]
        ], dtype=np.float32)
        mutant = cv2.warpAffine(seed, M, (height, width))
        return mutant
