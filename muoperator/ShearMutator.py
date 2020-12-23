import numpy as np
from muoperator.Mutator import Mutator
import cv2
import random


class ShearMutator(Mutator):
    """
        A mutator for image shear
    """

    def mutate(self, seed):
        """
            剪裁一个子区域并补边
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        height, width = seed.shape[:2]  # 获取图像的高和宽

        # 随机起点和终点
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)

        x2 = random.randint(0, width)
        y2 = random.randint(0, height)

        if x1 == x2:
            if x2 == width:
                x1 = x1 - 1
            else:
                x2 = x2 + 1

        if y1 == y2:
            if y2 == height:
                y1 = y1 - 1
            else:
                y2 = y2 + 1

        # 裁剪坐标为[y0:y1, x0:x1]
        cropped = seed[np.min([y1, y2]):np.max([y1, y2]), np.min([x1, x2]):np.max([x1, x2])]

        M = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        mutant = cv2.warpAffine(cropped, M, (height, width))

        return mutant
