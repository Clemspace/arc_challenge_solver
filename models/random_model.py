from arc_challenge_solver.models.base_model import ARCModel
import numpy as np
from typing import List, Dict, Tuple

class RandomModel(ARCModel):
    def __init__(self, train_pairs: List[Dict]):
        self.train_pairs = train_pairs
        self.max_grid_size = 30  # Assuming maximum grid size is 30x30
        self.num_colors = 10  # Assuming 10 possible colors/actions

    def train(self, train_pairs: List[Dict]) -> None:
        pass  # No training needed for random model

    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        # Generate random output with the same shape as the input grid
        output_shape = input_grid.shape
        return np.random.randint(0, self.num_colors, size=output_shape)
