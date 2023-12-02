import numpy as np
from typing import List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet


def tokenize(line: str, length: int = 1) -> Union[List[str], List[Tuple[str]]]:
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


class ROUGEL:
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
        hyp = tokenize(hyp)
        ref = tokenize(ref)
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


class CIDErD:
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
    

class Meteor:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.stemmer = PorterStemmer()
        self.wordnet = wordnet

    def basic_match(self, hypo_list, refer_list):
        word_match = []
        for i in range(len(hypo_list))[::-1]:
            for j in range(len(refer_list))[::-1]:
                if hypo_list[i][1] == refer_list[j][1]:
                    word_match.append((hypo_list[i][0], refer_list[j][0]))
                    hypo_list.pop(i)[1]
                    refer_list.pop(j)[1]
                    break
        return word_match, hypo_list, refer_list

    def stem_match(self, hypo_list, refer_list):
        stemmed_enum_list_hypo = [(word_idx, self.stemmer.stem(word)) for word_idx, word in hypo_list]
        stemmed_enum_list_ref = [(word_idx, self.stemmer.stem(word)) for word_idx, word in refer_list]
        word_match, unmat_hypo_idx, unmat_ref_idx = self.basic_match(stemmed_enum_list_hypo, stemmed_enum_list_ref)
        unmat_hypo_indices = set(idx[0] for idx in unmat_hypo_idx)
        unmat_ref_indices = set(idx[0] for idx in unmat_ref_idx)

        hypo_list = [(idx, word) for idx, word in hypo_list if idx not in unmat_hypo_indices]
        refer_list = [(idx, word) for idx, word in refer_list if idx not in unmat_ref_indices]

        return word_match, hypo_list, refer_list

    def get_synonyms(self, word):
        return set(lemma.name() for synset in self.wordnet.synsets(word) for lemma in synset.lemmas() if lemma.name().find("_") < 0)
    
    def wordnet_match(self, hypo_list, refer_list):
        word_match = []
        for i, (hypo_idx, hypo_word) in reversed(list(enumerate(hypo_list))):
            hypothesis_syns = self.get_synonyms(hypo_word)
            hypothesis_syns.add(hypo_word)

            match_found = any(refer_word == hypo_word or refer_word in hypothesis_syns for _, refer_word in reversed(refer_list))

            if match_found:
                word_match.append((hypo_idx, refer_list[-1][0]))
                hypo_list.pop(i)
                refer_list.pop()

        return word_match, hypo_list, refer_list

    def match(self, hypo_list, refer_list):
        exact_matches, hypo_list, refer_list = self.basic_match(hypo_list, refer_list)
        stem_matches, hypo_list, refer_list = self.stem_match(hypo_list, refer_list)
        wns_matches, hypo_list, refer_list = self.wordnet_match(hypo_list, refer_list)

        return sorted(exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0])

    def chunk_num(self, matches):
        i = 0
        chunks_num = 1
        while i < len(matches) - 1:
            if matches[i + 1][0] == matches[i][0] + 1 and matches[i + 1][1] == matches[i][1] + 1:
                i += 1
                continue
            i += 1
            chunks_num += 1
        return chunks_num

    def compute(self, reference, hypothesis):
        hypo_list = list(enumerate(hypothesis.lower().split()))
        refer_list = list(enumerate(reference.lower().split()))
        hypo_len = len(hypo_list)
        refer_len = len(refer_list)
        matches = self.match(hypo_list, refer_list)
        matches_num = len(matches)
        if matches_num == 0:
            return 0
        precision = matches_num / hypo_len
        recall = matches_num / refer_len
        fmean = precision * recall / (self.alpha * precision + (1 - self.alpha) * recall)
        chunk_num = self.chunk_num(matches)
        frag_frac = chunk_num / matches_num
        penalty = self.gamma * frag_frac ** self.beta
        meteor_score = (1 - penalty) * fmean
        return meteor_score

    def __call__(self, references, hypothesis):
        scores = [self.compute(reference, hypothesis) for reference in references]
        return max(scores)


if __name__ == '__main__':
    reference_sentence = "reference sentence"
    generated_sentence = "generated sentence"


    # TODO: Bool1020 - ROUGEL unit test code

    # TODO: zzc300 - CIDErD unit test code

    # Heathcliff-Zhao - Meteor unit test code
    from nltk.translate.meteor_score import meteor_score
    meteor_scorer = Meteor(alpha=0.9, beta=3, gamma=0.5)
    score = meteor_scorer([reference_sentence], generated_sentence)
    print("Meteor Score:", score)
    right_score = meteor_score([reference_sentence], generated_sentence)
    print("Right Meteor Score:", right_score)
    assert abs(score - right_score) < 1e-5