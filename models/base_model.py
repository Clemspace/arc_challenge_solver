from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple

class ARCModel(ABC):
    @abstractmethod
    def train(self, train_pairs: List[Dict]) -> None:
        pass

    @abstractmethod
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        pass
