from seed.generator.RandomGenerator import RandomGenerator
import numpy as np


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
