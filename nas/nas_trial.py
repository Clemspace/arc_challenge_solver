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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_num_epochs(batch_size, target_steps, num_samples):
    steps_per_epoch = num_samples / batch_size
    num_epochs = max(1, int(target_steps / steps_per_epoch))
    return num_epochs


def main():
    try:
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

        # Calculate num_epochs based on batch_size
        task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
        train_pairs = task.get_train_pairs()
        target_steps = 5000  # Adjust this based on your desired total computation
        num_samples = len(train_pairs)
        num_epochs = max(1, int(target_steps / (num_samples / batch_size)))
        
        logger.info(f"Calculated num_epochs: {num_epochs}")

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
        
        logger.info("Starting model training")
        model.train()
        
        logger.info("Evaluating model")
        avg_reward, perfect_solve_percentage = evaluate_model(model, task.get_eval_pairs())
        
        logger.info(f"Average evaluation score: {avg_reward}")
        logger.info(f"Perfect solve percentage: {perfect_solve_percentage}%")
        
        # Report the average reward as the final result
        nni.report_final_result(avg_reward)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        nni.report_final_result(float('-inf'))

if __name__ == "__main__":
    main()