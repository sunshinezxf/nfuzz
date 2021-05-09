# this file is responsible for assign mutation operators
import cv2
import numpy
import random
from muoperator.Mutator import Mutator
from muoperator.ScaleMutator import ScaleMutator
from muoperator.TranslationMutator import TranslationMutator
from muoperator.ShearMutator import ShearMutator
from muoperator.TransposeMutator import TransposeMutator
from muoperator.NoiseMutator import NoiseMutator
from muoperator.BlurMutator import BlurMutator
from muoperator.ContrastMutator import ContrastMutator
from keras.datasets import mnist

"""
    仿射变换其三个参数分别为:输入图像,变换矩阵,输出图像大小
    deepHunter中选择四种：平移、缩放、剪切、旋转
    仿射变换只能选择一次
"""
# alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
ALPHA = 0.02
BETA = 0.2
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def is_failed_test(I1):
    """
    判断是否是failed test
    :param I1:
    :return:
    """
    return False


def is_satisfied(seed, mutant):
    """
    判断mutant是否有意义
    :param seed:
    :param mutant:
    :return:
    """
    # print('is satisfied??',seed.shape)
    height, width = seed.shape[:2]  # 获取图像的高和宽
    l0, l_inf = 0, 0
    for row in range(height):  # 遍历高
        for col in range(width):
            p_seed = seed[row, col]
            p_mutant = mutant[row, col]
            if p_seed != p_mutant:  # 像素值发生改变
                l0 = l0 + 1

            minus = abs(int(p_seed) - int(p_mutant))
            if minus > l_inf:
                l_inf = minus  # 像素值发生的最大改变

    if l0 < ALPHA * height * width:
        if l_inf <= 255:
            return True
        else:
            return False
    else:
        if l_inf < 255 * BETA:
            return True
        else:
            return False


# 进行变换
def transform(state, seed):
    """
    所有的变异算子
    :param state:
    :param seed:
    :return: 变异后的图片
    """
    if state == 0:  # 缩放
        return ScaleMutator().mutate(seed)
    if state == 1:  # 平移
        return TranslationMutator().mutate(seed)
    if state == 2:  # 剪裁
        return ShearMutator().mutate(seed)
    if state == 3:  # 旋转
        return TransposeMutator().mutate(seed)
    if state == 4:  # 噪声
        return NoiseMutator().mutate(seed)
    if state == 5:  # 模糊
        return BlurMutator().mutate(seed)
    if state == 6:  # 对比度/亮度
        return ContrastMutator().mutate(seed)


def random_pick():
    """
    非仿射变换算子中随机选择一种算子进行变异
    :return:算子序号
    """
    s = random.randint(4, 6)
    return s


def random_pick_all():
    """
    在所有算子中随机选择一种算子进行变异
    :return:算子序号
    """
    s = random.randint(0, 6)
    return s


def image_mutate(try_num, seed, label):
    """
    deepHunter alg2 图像变异
    :param try_num: 最大尝试次数
    :param seed: 初始种子(单个图)
    :param label:该图片的标签
    :return: 变异成功的新种子或者原种子
    """
    state = 0
    s = 0
    I = I0 = I01 = seed
    I1 = None
    t = -1
    for i in range(try_num):
        if state == 0:  # 还没选过仿射
            t = random_pick_all()
            if t <= 3:  # 选了仿射变换
                state = 1
        else:
            t = random_pick()

        I1 = transform(t, I)

        if is_failed_test(I1):
            # todo:加入失败集，标签改变，可以单独抽出去。

            pass

        elif is_satisfied(I01, I1):
            if t > 4:
                state = 1
                I01 = transform(t, I0)
                return I1, label
    return I, label


def batch_mutate(batch):
    """
    返回变异完的一批种子
    :param batch:
    :return:
    """
    ret = []
    for img_tuple in batch:
        temp = image_mutate(3, img_tuple[0], img_tuple[1])
        ret.append(temp)
        # print(temp[0].shape, '------------------------')
        # print(temp[1])
    return ret
