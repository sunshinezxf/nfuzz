class Seed:
    """
    对种子进行封装，定义比较函数，使之可以在优先队列中使用
    """

    def __init__(self,image,label,probability=0):
        """
        封装种子element
        :param probability:被选择的概率
        """
        self.image=image
        self.label=label
        self.potential=probability

    def __lt__(self, other):
        """
        大根堆，优先选取probability大的种子
        :param other:
        :return:
        """
        if self.potential<other.potential:
            return False
        else:
            return True

    def val(self):
        return self.image, self.label