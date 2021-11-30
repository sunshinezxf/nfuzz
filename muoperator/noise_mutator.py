import numpy as np
from muoperator.Mutator import Mutator
import random


class NoiseMutator(Mutator):
    """
        A mutator for image noise
    """
    def mutate(self, seed):
        """
             添加椒盐噪声
            prob:噪声比例,0.1
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        prob = random.random() / 5
        mutant = np.zeros(seed.shape, np.uint8)
        thres = 1 - prob
        for i in range(seed.shape[0]):
            for j in range(seed.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    mutant[i][j] = 0
                elif rdn > thres:
                    mutant[i][j] = 255
                else:
                    mutant[i][j] = seed[i][j]
        return mutant
