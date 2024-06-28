import json
import os
import numpy as np
from typing import List, Dict
from .arc_task import ARCTask

class ARCDataLoader:
    @staticmethod
    def load_tasks(train_dir: str, eval_dir: str) -> ARCTask:
        train_pairs = ARCDataLoader._load_from_directory(train_dir)
        eval_pairs = ARCDataLoader._load_from_directory(eval_dir)
        return ARCTask(train_pairs, eval_pairs)

    @staticmethod
    def _load_from_directory(directory: str) -> List[Dict]:
        pairs = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    task = json.load(f)
                    pairs.extend(task['train'])
                    pairs.extend(task['test'])
        return pairs