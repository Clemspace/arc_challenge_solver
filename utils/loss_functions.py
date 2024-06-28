import numpy as np
class ARCLoss:
    @staticmethod
    def binary_loss(predicted: np.ndarray, expected: np.ndarray) -> float:
        return -1 if predicted.shape != expected.shape or not np.array_equal(predicted, expected) else 0

    @staticmethod
    def absolute_difference_loss(predicted: np.ndarray, expected: np.ndarray) -> float:
        if predicted.shape != expected.shape:
            return np.sum(expected.shape)  # Penalize incorrect shape predictions
        return np.sum(np.abs(predicted - expected))