import numpy as np
from typing import List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json


class CaptionMetric:
    def __init__(self):
        pass

    def compute(self, hyp: str, ref: str) -> float:
        pass

    def __call__(self, hyp: str, refs: List[str]) -> float:
        pass

    def tokenize(self, line: str, length: int = 1) -> Union[List[str], List[Tuple[str]]]:
        line = re.sub(r'[^\w\s]', ' ', line)
        line = line.lower()
        line = line.split()
        if length > 1:
            iteration = []
            for i in range(length):
                iteration.append(line[i: len(line)-length+i+1])
            res = []
            for item in zip(*iteration):
                res.append(item)
            return res
        elif length == 1:
            return line
        else:
            raise ValueError('length should be a positive integer')


class ROUGE(CaptionMetric):
    def __init__(self, N: int = 1):
        super(ROUGE, self).__init__()
        self.N = N

    def compute(self, hyp: str, ref: str) -> float:
        hyp = self.tokenize(hyp, self.N)
        ref = self.tokenize(ref, self.N)
        score = 0
        for token in ref:
            if token in hyp:
                score += 1
        return score / len(ref)

    def __call__(self, hyp: str, refs: List[str]) -> float:
        scores = 0
        for ref in refs:
            scores += self.compute(hyp, ref)
        return scores / len(refs)


class ROUGEL(CaptionMetric):
    def __init__(self, beta: float = 1.2):
        super().__init__()
        self.beta = beta

    def LCS(self, seq1: List[str], seq2: List[str]) -> List[str]:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                result.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        result.reverse()
        return result

    def compute(self, hyp: str, ref: str) -> float:
        hyp = self.tokenize(hyp)
        ref = self.tokenize(ref)
        lcs = len(self.LCS(hyp, ref))
        P_lcs = lcs / len(hyp)
        R_lcs = lcs / len(ref)
        F_lcs = ((1 + self.beta ** 2) * R_lcs * P_lcs) / (R_lcs + self.beta ** 2 * P_lcs)
        return F_lcs

    def __call__(self, hyp: str, refs: List[str]) -> float:
        scores = 0
        for ref in refs:
            scores += self.compute(hyp, ref)
        return scores / len(refs)


class CIDErD(CaptionMetric):
    def __init__(self, path_tfidf_weights: str = 'tfidf_weights.json'):
        super().__init__()
        with open(path_tfidf_weights, 'r') as f:
            tfidf_weights = json.load(f)
        self.tfidf_weights = tfidf_weights

    def _compute_tfidf(self, corpus: List[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(vocabulary=self.tfidf_weights)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return tfidf_matrix.toarray()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        return cosine_similarity([vec1], [vec2])[0][0]

    def __call__(self, hyp: str, refs: List[str]) -> float:
        hyp_tokens = self.tokenize(hyp)
        refs_tokens = [self.tokenize(ref) for ref in refs]
        corpus = [' '.join(hyp_tokens)] + [' '.join(ref) for ref in refs_tokens]
        tfidf_matrix = self._compute_tfidf(corpus)
        hyp_tfidf = tfidf_matrix[0]
        scores = [self._cosine_similarity(hyp_tfidf, ref_tfidf) for ref_tfidf in tfidf_matrix[1:]]
        return float(np.mean(scores))