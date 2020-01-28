import statistics


class SparsenessAnalyzer:

    @staticmethod
    def l1_norm(v):
        return sum(map(lambda x: abs(x), v))

    @staticmethod
    def l2_norm(v):
        return sum(map(lambda x: x ** 2, v)) ** 0.5

    @staticmethod
    def sparseness(v):
        # 0 is dense
        # 1 is sparse
        return ((len(v) ** 0.5) - (SparsenessAnalyzer.l1_norm(v) / SparsenessAnalyzer.l2_norm(v))) / ((len(v) ** 0.5) - 1)

    @staticmethod
    def avg_sparseness(vectors):
        return statistics.mean(map(SparsenessAnalyzer.sparseness, vectors))

    @staticmethod
    def median_sparseness(vectors):
        return statistics.median(map(SparsenessAnalyzer.sparseness, vectors))

