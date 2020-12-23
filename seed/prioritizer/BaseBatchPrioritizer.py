from seed.prioritizer.BatchPrioritizer import BatchPrioritizer


class BaseBatchPrioritizer(BatchPrioritizer):
    def probability(self, fuzzed_times, p_min, gamma):
        """
            P = max(1 - f(B) / γ, p_min)
        """
        P = 1 - fuzzed_times / gamma
        return max(P, p_min)
