from seed.generator.RandomGenerator import RandomGenerator
import numpy as np
import cv2


# 利用opencv的图像操作
# 算法先使用deepHunter的算法一次仿射变换多次像素值变换


class ImageRandomGenerator(RandomGenerator):
    """
        A image randomly seed generator
    """

    # shape of randomly generate seeds(numpy.ndarray), type is tuple
    shape = ()

    # constructor
    def __init__(self, shape):
        self.shape = shape

    def generate(self):
        """
            Randomly generate a seed for image
            :return：
                seed -- a randomly generated seed
        """
        return np.random.rand(*self.shape) * 255
