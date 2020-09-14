import abc


class Mutator(metaclass=abc.ABCMeta):
    """
        An interface for mutator
    """
    def mutate(self, seed):
        """
            Mutation seed
            :param:
                seed -- original seed
            :return：
                new_seed -- a mutant seed
        """
        pass
