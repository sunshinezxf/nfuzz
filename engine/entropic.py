import random
from myUtils import model_utils,muop_util


def assign_energy(t):
    """
    对种子t的energy进行赋值
    由local entropy H^t 度量
    :param t:
    :return:
    """


def rand_pick(corpus, sum_energy=1):
    """
    以energy值的大小为标准抽取seed
    把通过概率抽取某值的问题，变为某值落入区间的概率问题。
    :param corpus: 种子语料库
    :param sum_energy: corpus中种子的energy总和
    :return:
    """
    x = random.uniform(0, sum_energy)
    cum_prob = 0.0
    for seed in corpus:
        cum_prob += seed.energy
        if x < cum_prob:
            return seed


def cal_coverage(seed, model):
    """
    计算变异后的种子的覆盖率，判断是否提高了覆盖率
    :param seed:
    :param model:
    :return:
    """


def process(seed_corpus,model,max_loop=10):
    """
    Entropic主流程
    1.对于属于corpus的每个种子t的energy通过assign算法进行赋值
    2.对每个t的energy进行归一化
    3.以energy为概率从corpus中选取种子并进行mutate得到t'。
    4.如果t'是failed test则直接返回t'
    5.如果t'增加了覆盖率，则将t'加入corpus
    6.更新Y^it=Y^it+1,即对种子t进行fuzz产生的属于物种i的input的数量

    :param model:
    :param max_loop:最大循环次数or timeout
    :param seed_corpus:种子corpus todo:corpus的数据结构和deepHunter的如何统一
    :return:扩增后的seed corpus
    """

    for i in range(max_loop):
        total = 0
        for t in seed_corpus:
            assign_energy(t)  # 初始化每个种子t的energy
            total = total + t.energy

        # energy归一化
        for t in seed_corpus:
            t.energy = t.energy / total

        # 以energy为概率从corpus中选取种子并进行mutate
        seed = rand_pick(seed_corpus)
        mutant_t = muop_util.image_mutate(seed[0], seed[1])

        # 如果t'是failed test则直接返回t'
        if model_utils.is_failed_test(mutant_t,model):
            return mutant_t

        # 如果t'增加了覆盖率，则将t'加入corpus
        cal_coverage(mutant_t, model)

        # 更新Y^it=Y^it+1,即对种子t进行fuzz产生的属于物种i的input的数量