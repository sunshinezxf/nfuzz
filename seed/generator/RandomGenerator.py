import abc


class RandomGenerator(metaclass=abc.ABCMeta):
    """
        An interface for randomly generate seed
    """
    def random_generate(self):
        """
            Randomly generate a seed
            :return：
                seed -- a randomly generated seed
        """
        pass
