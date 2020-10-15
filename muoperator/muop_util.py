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

"""
    仿射变换其三个参数分别为:输入图像,变换矩阵,输出图像大小
    deepHunter中选择四种：平移、缩放、剪切、旋转
    仿射变换只能选择一次
"""
# alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
ALPHA = 0.02
BETA = 0.2


# 判断mutant是否有意义
def is_satisfied(seed, mutant):
    height, width = seed.shape[:2]  # 获取图像的高和宽
    l0, l_inf = 0, 0
    for row in range(height):  # 遍历高
        for col in range(width):
            p_seed = seed[row, col]
            p_mutant = mutant[row, col]
            if p_seed != p_mutant:  # 像素值发生改变
                l0 = l0 + 1

            minus = abs(p_seed - p_mutant)
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


def transform(state, seed):
    if state == 0:  # 缩放
        return ScaleMutator().mutate(seed)
    if state == 1:  # 平移
        return TranslationMutator.mutate(seed)
    if state == 2:  # 剪裁
        return ShearMutator.mutate(seed)
    if state == 3:  # 旋转
        return TransposeMutator.mutate(seed)
    if state == 4:  # 噪声
        return NoiseMutator.mutate(seed)
    if state == 5:  # 模糊
        return BlurMutator.mutate(seed)
    if state == 6:  # 对比度/亮度
        return ContrastMutator.mutate(seed)


def random_pick(state):
    if state == 0:  # 可用选择一次仿射
        s = random.randint(0, 6)
        return s
    else:
        s = random.randint(4, 6)
        return s

# deepHunter alg2
def image_mutate(try_num, seed):
    """
    :param try_num: 最大尝试次数
    :param seed: 初始种子(单个图)
    :return: 变异成功的新种子或者原种子
    """
    state = 0
    I = I0 = I01 = seed
    I1 = None
    t = -1
    for i in range(try_num):
        if state == 0:
            t = random_pick(state)
        else:
            t = random_pick(state)

        I1 = transform(t, I)

        if is_satisfied(I01, I1):
            if t > 4:
                state = 1
                I01 = transform(t, I0)
                return I1
    return I

