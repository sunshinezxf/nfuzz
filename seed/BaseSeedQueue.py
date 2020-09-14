import queue
import numpy

from seed.SeedQueue import SeedQueue


class BaseSeedQueue(SeedQueue):
    """
        A base queue for seed operate
    """

    # The queue to store seeds
    queue = queue.Queue()

    def push(self, seed):
        """
            Push the seed into queue, the type of seed must be 'numpy.ndarray'
            :param：
                seed -- seed of type 'numpy.ndarray'
            :except：
                TypeError -- type not match
        """
        if not isinstance(seed, numpy.ndarray):
            raise TypeError("The type of seed must be 'numpy.ndarray'.")
        self.queue.put(seed)

    def pop(self):
        """
            Pop the seed from queue and return it
            :return：
                seed -- seed of type 'numpy.ndarray'
            :except：
                StopIteration -- The queue is empty
        """
        if self.queue.empty():
            raise StopIteration("The queue is empty.")
        return self.queue.get()

    def empty(self):
        """
            Whether the queue is empty
            :return：
                true | false
        """
        return self.queue.empty()
