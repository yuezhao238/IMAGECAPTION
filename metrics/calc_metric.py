from metrics.metrics import *
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

class metric_logger:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.rougel_scorer = ROUGEL(beta=1.2)
        self.cider_scorer = CIDErD()
        self.meteor_scorer = Meteor(alpha=0.9, beta=3, gamma=0.5)
        self.bleu_scorer = corpus_bleu

    def _calc(self, refs, cands):
        rouge_l = np.mean([self.rougel_scorer(cands[i], refs[i]) for i in range(len(refs))])
        cider = self.cider_scorer.compute_score(cands, refs)[0]['CIDEr_d'].item()
        meteor = np.mean([self.meteor_scorer(refs[i], cands[i]) for i in range(len(refs))])
        bleu = self.bleu_scorer(refs, cands)
        return rouge_l, cider, meteor, bleu

    def log(self, refs, cands, step):
        rouge_l, cider, meteor, bleu = self._calc(refs, cands)
        self.wandb_logger.log({"rouge_l": rouge_l, "cider": cider, "meteor": meteor, "bleu": bleu, "step": step})
        return rouge_l, cider, meteor, bleu
        
    
