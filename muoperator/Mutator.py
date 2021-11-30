import abc
from myUtils import muop_util


class Mutator(metaclass=abc.ABCMeta):
    """
        An interface for mutator
    """
    # alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
    ALPHA = 0.02
    BETA = 0.2

    def __init__(self):
        pass

    @abc.abstractmethod
    def mutate(self, batch):
        """
            对一批种子进行变异
            :param:
                seed -- original seed
            :return：
                new_seed -- a mutant seed
        """

        return muop_util.batch_mutate(batch)


    """仿射变换其三个参数分别为:输入图像,变换矩阵,输出图像大小
    deepHunter中选择四种：平移、缩放、剪切、旋转
    """















