import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Dict
from collections import defaultdict, Counter
from typing import Any, Callable, Mapping, Union
import torch
from torch import Tensor



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
    def __init__(self, n: int = 4, sigma: float = 6.0, scale: float = 10.0):
        self.n = n
        self.sigma = sigma
        self.scale = scale

    def compute_score(self, candidates: list[str], 
                      references: list[list[str]], 
                      return_all_scores: bool = True, 
                      return_tfidf: bool = False) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
        # 预处理
        process_cands, process_refes = self._prepare_data(candidates, references)
        # 计算打分
        score = self._compute(process_cands, process_refes, return_all_scores, return_tfidf)
        return score

    def _prepare_data(self, candidates: list[str], references: list[list[str]]) -> tuple[list, list]:
        # 候选文本数量与参考文本数量相同
        if len(candidates) != len(references):
            raise ValueError(f"Invalid number of candidates and references. (found {len(candidates)=} != {len(references)=})")
        new_process_refes = [[self._process_sentence(ref) for ref in refs] for refs in references]
        new_process_cands = [self._process_sentence(cand) for cand in candidates]
        return new_process_cands, new_process_refes

    def _compute(self, process_cands: list[Counter], 
                 process_refes: list[list[Counter]], 
                 return_all_scores: bool, 
                 return_tfidf: bool) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
        # 至少需要两个候选文本及其对应的参考文本
        if len(process_cands) <= 1:
            raise ValueError(f"CIDEr-D metric does not support less than 2 candidates with 2 references. (found {len(process_cands)} candidates, but expected > 1)")
        # 计算参考文本中n-gram的文档频率
        doc_frequencies = self._compute_doc_freq(process_refes)
        # 候选文本的数量大于等于任何n-gram的最大文档频率
        assert len(process_cands) >= max(doc_frequencies.values()), "Sanity check failed."
        log_refs = np.log(float(len(process_refes)))
        # 计算每个候选文本与其对应参考文本之间的CIDEr-D评分
        cider_scores, tfidf_lst = self._compute_cider(process_cands, process_refes, doc_frequencies, log_refs)
        cider_score = cider_scores.mean()
        cider_scores = torch.from_numpy(cider_scores)
        cider_score = torch.as_tensor(cider_score, dtype=torch.float64)
        if return_all_scores:
            cider_outs_corpus = {"CIDEr_d": cider_score}
            cider_outs_sents = {"CIDEr_d": cider_scores}
            if return_tfidf:
                cider_outs_sents["tfidf_lst"] = tfidf_lst
            cider_outs = cider_outs_corpus, cider_outs_sents
            return cider_outs
        else:
            return cider_score

    def _process_sentence(self, sentence: str) -> Counter[tuple[str, ...]]:
        words = tokenize(sentence)
        # ngram计数
        ngram_counter = Counter()
        # 生成n-gram
        for k in range(1, self.n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i : i + k])
                ngram_counter[ngram] += 1
        return ngram_counter

    def _compute_doc_freq(self, process_refes: list[list[Counter]]) -> Counter[tuple[str, ...]]:
        doc_frequencies = Counter()
        for refs in process_refes:
            # 创建一个集合 all_refs_ngrams，包含了当前参考文本集（refs）中所有的唯一 n-gram
            all_refs_ngrams = set(ngram for ref in refs for ngram in ref.keys())
            for ngram in all_refs_ngrams:
                doc_frequencies[ngram] += 1 # 代表该n-gram在多少个不同的参考文本中出现过
        return doc_frequencies

    def _compute_cider(self, process_cands: list[Counter], 
                       process_refes: list[list[Counter]], 
                       doc_frequencies: Union[Counter[tuple], Callable[[tuple], int]], 
                       log_refs: float) -> tuple[np.ndarray, list[tuple[list, list]]]:
        scores = np.empty((len(process_cands),))
        tfidf_lst = []
        # 候选文本的n-gram计数转换为向量形式，同时计算其范数和长度
        for i, (cand, refs) in enumerate(zip(process_cands, process_refes)):
            vec, norm, length = self._counter_to_vec(cand, log_refs, doc_frequencies)
            ngrams_scores = np.zeros((len(refs), self.n))
            vec_refs = []
            # 将每个参考文本n-gram计数转换为向量，并计算其范数和长度
            for j, ref in enumerate(refs):
                vec_ref, norm_ref, length_ref = self._counter_to_vec(ref, log_refs, doc_frequencies)
                vec_refs.append(vec_ref)
                ngrams_scores[j] = self._similarity(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = ngrams_scores.sum(axis=0).mean() / len(refs)
            scores[i] = score_avg
            tfidf_lst.append((vec, vec_refs))
        scores = scores * self.scale
        return scores, tfidf_lst

    def _counter_to_vec(self, counters: dict[tuple, int], 
                        log_refs: float, 
                        doc_frequencies: Union[Mapping[tuple, int], Callable[[tuple], int]]) -> tuple[list[defaultdict], np.ndarray, int]:
        vec = [defaultdict(float) for _ in range(self.n)] # 存储每个n-gram的TF-IDF值
        length = 0  # 文本的长度
        norm = np.zeros((self.n,))  # 存储每个n-gram向量的范数

        for ngram, term_freq in counters.items():
            # 确定文档频率是通过映射还是函数获取的
            if isinstance(doc_frequencies, Mapping):
                count = doc_frequencies[ngram]
            else:
                count = doc_frequencies(ngram)

            log_df = np.log(max(1.0, count))
            # 根据n-gram的长度确定当前n-gram的索引
            cur_n = len(ngram) - 1
            # 计算n-gram的TF-IDF值并存储在 vec 中
            vec[cur_n][ngram] = float(term_freq) * (log_refs - log_df)
            # 更新 norm 数组，表示每个n-gram向量的范数
            norm[cur_n] += pow(vec[cur_n][ngram], 2)

            if cur_n == 1:
                length += term_freq

        norm = np.sqrt(norm)
        return vec, norm, length

    def _similarity(self, cand_vec: list[defaultdict], ref_vec: list[defaultdict], cand_norm: np.ndarray, ref_norm: np.ndarray, cand_len: int, ref_len: int) -> np.ndarray:
        delta = int(cand_len - ref_len)
        similarities = np.zeros((self.n,))

        # 外层循环遍历不同的n-gram长度
        for ni in range(self.n):
            # 内层循环遍历候选文本的n-gram及其计数
            for ngram, count in cand_vec[ni].items():
                # 相似度计算基于候选文本和参考文本中相同n-gram的计数
                similarities[ni] += min(count, ref_vec[ni][ngram]) * ref_vec[ni][ngram]
            # 如果n-gram向量的范数不为零，则通过候选文本和参考文本的范数来标准化相似度
            if (cand_norm[ni] != 0) and (ref_norm[ni] != 0):
                similarities[ni] /= cand_norm[ni] * ref_norm[ni]
            # 长度惩罚因子
            similarities[ni] *= np.e ** (-(delta**2) / (2 * self.sigma**2))

        return similarities


    

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
    from rouge import Rouge
    rougel_scorer = ROUGEL(beta=1.2)
    score = rougel_scorer(generated_sentence, [reference_sentence])
    print("ROUGE-L Score:", score)
    rouge = Rouge()
    right_score = rouge.get_scores(generated_sentence, reference_sentence)[0]['rouge-l']['f']
    print("Right ROUGE-L Score:", right_score)
    assert abs(score - right_score) < 1e-5

    # TODO: zzc300 - CIDErD unit test code
    from aac_metrics.functional import cider_d
    candidates : list[str] = ["a man is speaking", "rain falls"]
    references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]
    ciderd = CIDErD()
    score = ciderd.compute_score(candidates, references)
    final_score = round(score[0]['CIDEr_d'].item(), 6)
    print("CIDErD Score:", final_score)
    corpus_scores, sents_scores = cider_d(candidates, references)
    right_score = corpus_scores['cider_d'].item()
    print("Right CIDErD Score:", right_score)
    assert abs(final_score - right_score) < 1e-5

    # Heathcliff-Zhao - Meteor unit test code
    from nltk.translate.meteor_score import meteor_score
    meteor_scorer = Meteor(alpha=0.9, beta=3, gamma=0.5)
    score = meteor_scorer([reference_sentence], generated_sentence)
    print("Meteor Score:", score)
    right_score = meteor_score([reference_sentence], generated_sentence)
    print("Right Meteor Score:", right_score)
    assert abs(score - right_score) < 1e-5