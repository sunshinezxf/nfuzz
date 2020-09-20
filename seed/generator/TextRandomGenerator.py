from seed.generator.RandomGenerator import RandomGenerator
import random
import string
import numpy as np
import hanlp

# @software{hanlp2,
#   author = {Han He},
#   title = {{HanLP: Han Language Processing}},
#   year = {2020},
#   url = {https://github.com/hankcs/HanLP},
# }
# 由于对python的熟悉度还不高，先写思路
# 数据源暂时不管先用简单的磁盘文本作测试，后续使用数据库相关
# 生成方面借助HanLP进行分词和词性标注
# 假设语料具有相似的结构，变异算法为将语料的各部位随机互换


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
        tokenizer = hanlp.load()
        s = ''.join(random.sample(string.ascii_letters + string.digits, self.length))
        return np.array(list(s))
