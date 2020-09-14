import numpy as np

from muoperator.Mutator import Mutator
from PIL import Image


class TransposeMutator(Mutator):
    """
        A mutator for image transpose
    """
    def mutate(self, seed):
        """
            Mutation image seed, random rotation 0-360 degrees
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        pil_img = Image.fromarray(seed, mode='RGB')
        new_img = pil_img.rotate(np.random.randint(0, 360))
        return np.array(new_img)
