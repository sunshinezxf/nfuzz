

class Seed:
    """
    对种子进行封装
    """

    def __init__(self,seed,probability):
        """
        封装种子element
        :param seed: element type is '{fuzzedTimes, batch}'
        :param probability:
        """
        self.seed=seed
        self.probability=probability

    def __lt__(self, other):
        """
        大根堆
        :param other:
        :return:
        """
        if self.probability<other.probability:
            return False
        else:
            return True

    def val(self):
        return self.seed