from muoperator.Mutator import Mutator
import cv2
import random


class ContrastMutator(Mutator):
    """
        A mutator for image contrast
    """
    def mutate(self, seed):
        """
            改变图像对比度/亮度
            :param:
                seed -- original image seed
            :return：
                new_seed -- a mutant image seed
        """
        brightness = random.randint(1, 100)
        contrast = random.random()
        # 图像混合( cv2.addWeighted() )
        # 这也是图像添加，但是对图像赋予不同的权重，使得它具有混合感或透明感。
        mutant = cv2.addWeighted(seed, contrast, seed, 0, brightness)
        return mutant
