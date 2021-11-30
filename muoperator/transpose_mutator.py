from muoperator.Mutator import Mutator
import cv2
import random


class TransposeMutator(Mutator):
    """
        A mutator for image transpose
    """
    def mutate(self,seed):
        """
            Mutation image seed, random rotation 0-360 degrees
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        # print(seed.shape)
        height, width = seed.shape[:2]  # 获取图像的高和宽
        center = (width / 2, height / 2)  # 默认中心
        angle = random.randint(0, 360)
        scale=1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        mutant = cv2.warpAffine(seed, M, (height, width))
        return mutant
