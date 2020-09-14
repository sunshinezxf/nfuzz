import abc


class BatchPrioritizer(metaclass=abc.ABCMeta):
    """
        Interface for calculate the possibility of element
    """
    @abc.abstractmethod
    def probability(self, fuzzed_times, p_min, gamma):
        """
            Calculate the possibility of element
            :param
                fuzzed_times -- represents how many times the batch B has been fuzzed
                p_min --  the minimum probability
                gamma -- weight
            :return
                probability -- probality of batch be selected
        """
        pass
