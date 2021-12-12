

class Batch:
    """
    存放一批Seed
    """
    def __init__(self,batch,fuzzed_times):
        """
        封装种子element
        :param batch: seed列表[]
        """
        self.batch=batch
        self.fuzzed_times=fuzzed_times
        self.probability=self.probability(fuzzed_times)

    def __lt__(self, other):
        """
        大根堆，优先选取probability大的种子
        :param other:
        :return:
        """
        if self.probability<other.probability:
            return False
        else:
            return True

    def val(self):
        return self.batch

    def probability(self, fuzzed_times, p_min=0.1, gamma=1):
        """
            P = max(1 - f(B) / γ, p_min)
        """
        P = 1 - fuzzed_times / gamma
        return max(P, p_min)