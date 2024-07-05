import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arc_challenge_solver.config.config import TRAIN_DIR, EVAL_DIR
from arc_challenge_solver.data.arc_dataloader import ARCDataLoader
from arc_challenge_solver.models.ppo_model import PPOModel
from arc_challenge_solver.utils.metrics import calculate_adjusted_score, evaluate_model
import nni
import numpy as np
import torch

def calculate_num_epochs(batch_size, target_steps, num_samples):
    steps_per_epoch = num_samples / batch_size
    num_epochs = max(1, int(target_steps / steps_per_epoch))
    return num_epochs

def main():
    # Receive parameters from NNI
    params = nni.get_next_parameter()
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_layers = params['num_layers']
    hidden_dim = params['hidden_dim']
    dropout_rate = params['dropout_rate']
    optimizer_name = params['optimizer']
    activation_name = params['activation']
    use_skip_connections = params['skip_connections']
    weight_decay = params['weight_decay']

    # Load training pairs
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    train_pairs = task.get_train_pairs()
    
    # Calculate num_epochs based on batch_size
    target_steps = 1000  # Adjust this based on your desired total computation
    num_samples = len(train_pairs)
    num_epochs = calculate_num_epochs(batch_size, target_steps, num_samples)
    
    print(f"Calculated num_epochs: {num_epochs}")

    try:
        model = PPOModel(
            train_pairs=train_pairs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            activation_name=activation_name,
            use_skip_connections=use_skip_connections,
            weight_decay=weight_decay,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train the model
        model.train()
        
        # Evaluate the model and report the final result to NNI
        avg_reward = evaluate_model(model, task.get_eval_pairs())
        print(f"Final evaluation score: {avg_reward}")
        nni.report_final_result(avg_reward)        

        # Save the model state for potential later analysis
        model_state = {
            'model_state_dict': model.policy.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'score': avg_reward,
            'params': params
        }
        
        # The NNI trial ID can be used to identify this particular run
        trial_id = nni.get_trial_id()
        save_dir = os.path.join('nas_results', trial_id)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_state, os.path.join(save_dir, 'model_state.pth'))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())  # This will print the full traceback
        nni.report_final_result(float('-inf'))




if __name__ == "__main__":
    main()