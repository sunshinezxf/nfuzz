import numpy as np
from muoperator.Mutator import Mutator
import cv2
import random


class BlurMutator(Mutator):
    """
        A mutator for image blur
    """
    def mutate(self, seed):
        """
            高斯模糊。中心点领域
             https://blog.csdn.net/wuqindeyunque/article/details/103694900?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
            这个函数可以根据ksize和sigma求出对应的高斯核，计算公式为
            sigma = 0.3*((ksize-1)*0.5-1)+0.8
            当ksize=3时，sigma=0.8
            当ksize=5时，sigma为1.1.
            kszie,sigma越大越模糊
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        ksize = random.randint(1, 5)
        sigma = random.random() * 10
        mutant = cv2.GaussianBlur(seed, (ksize, ksize), sigma)
        return mutant
