from queue import PriorityQueue


class PriorityPool:

    """
    优先级batchPool，可以按照给定的优先级从pool中获得种子batch
    """

    def __init__(self):
        self.queue=PriorityQueue()

    def push(self,seed):
        """
        往pool中加入种子
        :param seed:
        :return:
        """
        self.queue.put(seed)

    def pop(self):
        """
        返回优先级最高的种子
        :return:
        """
        if self.queue.empty():
            raise StopIteration("The queue is empty.")
        return self.queue.get().val()

    def empty(self):
        """
        判断是否为空
        :return:
        """
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()