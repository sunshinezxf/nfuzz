from seed.generator.RandomGenerator import RandomGenerator
import random
import string
import numpy as np


class TextRandomGenerator(RandomGenerator):
    """
        A text randomly seed generator
    """

    # length of randomly generate seeds(numpy.ndarray)
    # default length is 10
    length = 10

    # constructor
    def __init__(self, length):
        self.length = length

    def generate(self):
        """
            Randomly generate a seed for text
            :return：
                seed -- a randomly generated seed
        """
        s = ''.join(random.sample(string.ascii_letters + string.digits, self.length))
        return np.array(list(s))

