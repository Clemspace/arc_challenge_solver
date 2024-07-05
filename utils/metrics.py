import numpy as np

def calculate_reward(predicted_output, true_output):
    # Full task accuracy (1 if completely correct, 0 otherwise)
    full_task_accuracy = 1 if np.array_equal(predicted_output, true_output) else 0
    
    # Pixel-wise accuracy
    total_pixels = true_output.size
    correct_pixels = np.sum(predicted_output == true_output)
    pixel_accuracy = correct_pixels / total_pixels
    
    # Composite reward
    reward = (full_task_accuracy + pixel_accuracy) / 2
    
    return reward

def proportion_correct(predicted: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the proportion of correctly predicted values.
    
    Args:
    predicted (np.ndarray): The predicted output grid.
    expected (np.ndarray): The expected output grid.
    
    Returns:
    float: The proportion of correctly predicted values (0.0 to 1.0).
    """
    print(f"Proportion correct - Predicted shape: {predicted.shape}, Target shape: {expected.shape}")  # Debug statement

    if predicted.shape != expected.shape:
        raise ValueError("Predicted and expected arrays must have the same shape.")
    
    total_elements = np.prod(expected.shape)
    correct_elements = np.sum(predicted == expected)
    
    return correct_elements / total_elements


def calculate_adjusted_score(predicted, target):
    predicted_shape = predicted.shape
    target_shape = target.shape
    
    # Calculate the size prediction accuracy
    size_accuracy = 1 - (abs(predicted_shape[0] - target_shape[0]) + abs(predicted_shape[1] - target_shape[1])) / (target_shape[0] + target_shape[1])
    
    # If shapes do not match, calculate a penalty
    if predicted_shape != target_shape:
        min_shape = (
            min(predicted_shape[0], target_shape[0]),
            min(predicted_shape[1], target_shape[1])
        )
        # Crop both arrays to the minimum shape
        predicted_cropped = predicted[:min_shape[0], :min_shape[1]]
        target_cropped = target[:min_shape[0], :min_shape[1]]
        
        # Calculate the proportion correct on the cropped arrays
        correct_proportion = proportion_correct(predicted_cropped, target_cropped)
        
        # Apply a penalty based on the difference in sizes
        size_penalty = 1 - size_accuracy
        
        adjusted_score = correct_proportion * (1 - size_penalty)
    else:
        # If shapes match, calculate the proportion correct
        adjusted_score = proportion_correct(predicted, target)
    
    return adjusted_score * size_accuracy

def evaluate_model_old(model, eval_pairs):
    scores = []
    size_accuracies = []
    for pair in eval_pairs:
        input_grid = np.array(pair['input'])
        expected_output = np.array(pair['output'])
        predicted_output = model.predict(input_grid)
        
        score = calculate_adjusted_score(predicted_output, expected_output)
        scores.append(score)
        
        # Calculate size accuracy
        size_accuracy = 1 - (abs(predicted_output.shape[0] - expected_output.shape[0]) + 
                             abs(predicted_output.shape[1] - expected_output.shape[1])) / (
                             expected_output.shape[0] + expected_output.shape[1])
        size_accuracies.append(size_accuracy)
    
    return np.mean(scores), np.mean(size_accuracies)

def evaluate_model(model, eval_pairs):
    rewards = []
    for pair in eval_pairs:
        input_grid = np.array(pair['input'])
        true_output = np.array(pair['output'])
        predicted_output = model.predict(input_grid)
        reward = calculate_reward(predicted_output, true_output)
        rewards.append(reward)
    
    avg_reward = np.mean(rewards)
    return avg_reward