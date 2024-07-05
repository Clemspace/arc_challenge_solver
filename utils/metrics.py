import numpy as np
def calculate_reward(predicted_output, true_output):
    # Ensure both outputs are numpy arrays
    predicted_output = np.array(predicted_output)
    true_output = np.array(true_output)
    
    # Check if shapes match
    if predicted_output.shape != true_output.shape:
        # If shapes don't match, we'll compare the overlapping part
        min_height = min(predicted_output.shape[0], true_output.shape[0])
        min_width = min(predicted_output.shape[1], true_output.shape[1])
        
        predicted_output = predicted_output[:min_height, :min_width]
        true_output = true_output[:min_height, :min_width]
    
    # Ensure both arrays have the same dtype
    dtype = np.result_type(predicted_output.dtype, true_output.dtype)
    predicted_output = predicted_output.astype(dtype)
    true_output = true_output.astype(dtype)
    
    total_pixels = np.prod(true_output.shape)
    correct_pixels = np.sum(predicted_output == true_output)
    
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # Size accuracy
    size_accuracy = 1 - (abs(predicted_output.shape[0] - true_output.shape[0]) + 
                         abs(predicted_output.shape[1] - true_output.shape[1])) / (
                         true_output.shape[0] + true_output.shape[1] + 1e-8)
    
    # Combine pixel accuracy and size accuracy
    reward = (pixel_accuracy + size_accuracy) / 2
    
    return reward

def evaluate_model(model, eval_pairs):
    scores = []
    for pair in eval_pairs:
        input_grid = np.array(pair['input'])
        true_output = np.array(pair['output'])
        predicted_output, predicted_size = model.predict(input_grid)
        
        # Ensure predicted_output matches the predicted size
        predicted_output = predicted_output[:predicted_size[0], :predicted_size[1]]
        
        try:
            score = calculate_reward(predicted_output, true_output)
            scores.append(score)
        except Exception as e:
            print(f"Error calculating reward: {e}")
            print(f"Predicted output shape: {predicted_output.shape}")
            print(f"True output shape: {true_output.shape}")
            scores.append(0)  # Append a zero score for this pair
    
    avg_score = np.mean(scores)
    print(f"Average evaluation score: {avg_score}")
    return avg_score

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

