from seed.generator.RandomGenerator import RandomGenerator
import numpy as np
import os
import cv2
from keras.datasets import mnist

class ImageRandomGenerator(RandomGenerator):
    """
        A image randomly seed generator
    """

    # shape of randomly generate seeds(numpy.ndarray), type is tuple
    shape = ()

    # constructor
    def __init__(self, shape):
        self.shape = shape

    def random_generate(self):
        """
            Randomly generate a seed for image
            :return：
                seed -- a randomly generated seed
        """
        randomByteArray = bytearray(os.urandom(120000))
        flatNumpyArray = np.array(randomByteArray)

        bgrImage = flatNumpyArray.reshape(self.shape)
        # grayImage = flatNumpyArray.reshape(300, 400) # 灰度

        return bgrImage

    
    def generate(self):
        """
        导入现有的种子图片
        :return:
            种子图片集
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train
