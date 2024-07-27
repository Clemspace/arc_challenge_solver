import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arc_challenge_solver.config.config import TRAIN_DIR, EVAL_DIR
from arc_challenge_solver.data.arc_dataloader import ARCDataLoader
from arc_challenge_solver.models.ppo_model import PPOModel
from arc_challenge_solver.utils.metrics import calculate_adjusted_score, evaluate_model
from arc_challenge_solver.utils.data_augmentation import augment_data
from arc_challenge_solver.utils.curriculum_learning import get_curriculum
import nni
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Add this import


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        params = nni.get_next_parameter()
        
        task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
        train_pairs = task.get_train_pairs()
        
        # Implement curriculum learning
        curriculum = get_curriculum(train_pairs)
        
        # Reduce batch size
        batch_size = min(params['batch_size'], 16)  # Limit batch size to 32
        
        model = PPOModel(
            train_pairs=train_pairs,
            hidden_dim=params['hidden_dim'],
            learning_rate=params['learning_rate'],
            batch_size=batch_size,
            num_epochs=3000,  # Increased number of epochs
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            optimizer_name=params['optimizer'],
            weight_decay=params['weight_decay'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(model.optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        # Early stopping
        best_reward = 0
        patience = 20
        patience_counter = 0
        
        # Training with curriculum learning
        for stage, stage_tasks in enumerate(curriculum):
            logger.info(f"Starting curriculum stage {stage + 1}/{len(curriculum)}")
            model.train_pairs = stage_tasks
            model.preprocess_data()
            
            for epoch in range(200):  # Train for 200 epochs per stage
                try:
                    loss, reward = model.train_epoch()
                    
                    # Report intermediate result
                    nni.report_intermediate_result(reward)
                    
                    logger.info(f"Stage {stage + 1}, Epoch {epoch + 1}: Loss = {loss:.4f}, Reward = {reward:.4f}")

                    # Evaluate after each epoch
                    avg_reward, perfect_solve_percentage = evaluate_model(model, task.get_eval_pairs())
                    logger.info(f"Evaluation - Avg reward: {avg_reward}, Perfect solves: {perfect_solve_percentage}%")

                    # Learning rate scheduling
                    scheduler.step(avg_reward)

                    # Early stopping
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("WARNING: out of memory, skipping epoch")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e
        
        # Final evaluation
        avg_reward, perfect_solve_percentage = evaluate_model(model, task.get_eval_pairs())
        logger.info(f"Final - Avg reward: {avg_reward}, Perfect solves: {perfect_solve_percentage}%")
        
        nni.report_final_result(avg_reward)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        nni.report_final_result(float('-inf'))

if __name__ == "__main__":
    main()