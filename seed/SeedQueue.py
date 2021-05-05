import abc


class SeedQueue(metaclass=abc.ABCMeta):
    """
        An interface for operate of seed_queue
    """



    def push(self, seed):
        """
            Push the seed into queue, the type of seed must be 'numpy.ndarray'
            :param：
                seed -- seed of type 'numpy.ndarray'
            :except：
                TypeError -- type not match
        """
        pass

    def pop(self):
        """
            Pop the seed from queue and return it
            :return：
                seed -- seed of type 'numpy.ndarray'
            :except：
                StopIteration -- The queue is empty
        """
        pass

    def empty(self):
        """
            Whether the queue is empty
            :return：
                true | false
        """
        pass
