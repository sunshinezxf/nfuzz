from seed.prioritizer.BatchPrioritizer import BatchPrioritizer
import math
import numpy

class EntropyBatchPrioritizer(BatchPrioritizer):
    def probability(self, fuzzed_times, p_min, gamma):
        """
            根据熵值分配能量。
            D是输入空间。模糊化是从D中抽取N个输入并进行替换的随机过程。
            把D分为S个子空间，代表S个物种。输入的种类是根据输入执行的动态程序属性定义的。例如，输入Xn∈D的每个分支都可以被识别为一个物种。新物种的发现相应地增加了分支覆盖率。

        """

        pass
