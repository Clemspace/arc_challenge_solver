import numpy as np
from typing import List, Dict

class ARCTask:
    def __init__(self, train_pairs: List[Dict], eval_pairs: List[Dict]):
        self.train_pairs = train_pairs
        self.eval_pairs = eval_pairs

    def get_train_pairs(self):
        return self.train_pairs

    def get_eval_pairs(self):
        return self.eval_pairs