import matplotlib.pyplot as plt
from data.arc_task import ARCTask
from utils.loss_functions import ARCLoss
from utils.visualizer import ARCVisualizer
from tqdm import tqdm
from models.base_model import ARCModel
import numpy as np
from models.ppo_model import PPOModel
class ARCExperiment:
    def __init__(self, task: ARCTask, model: ARCModel, visualize: bool = False):
        self.task = task
        self.model = model
        self.visualize = visualize
        self.eval_losses = []
        self.eval_rewards = []

    def run(self):
        # This method is now empty as we're training PPO separately
        pass

    def evaluate(self):
        binary_losses = []
        abs_diff_losses = []
        for pair in tqdm(self.task.eval_pairs, desc="Evaluating"):
            input_grid = np.array(pair['input'])
            expected_output = np.array(pair['output'])
            predicted_output = self.model.predict(input_grid)
            
            binary_loss = ARCLoss.binary_loss(predicted_output, expected_output)
            abs_diff_loss = ARCLoss.absolute_difference_loss(predicted_output, expected_output)
            
            binary_losses.append(binary_loss)
            abs_diff_losses.append(abs_diff_loss)
            
            self.eval_losses.append(abs_diff_loss)
            self.eval_rewards.append(-abs_diff_loss)  # Negative loss as reward
            
            if self.visualize:
                ARCVisualizer.visualize_comparison(input_grid, expected_output, predicted_output, binary_loss, abs_diff_loss)
        
        accuracy = (len(binary_losses) + sum(binary_losses)) / len(binary_losses)
        avg_abs_diff_loss = np.mean(abs_diff_losses)
        
        self.plot_results()
        
        return accuracy, avg_abs_diff_loss

    def plot_results(self):
        plt.figure(figsize=(15, 10))
        
        if isinstance(self.model, PPOModel):
            # Training plots for PPOModel
            plt.subplot(2, 2, 1)
            plt.plot(self.model.training_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(2, 2, 2)
            plt.plot(self.model.training_rewards)
            plt.title('Training Reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')

        # Evaluation plots
        plt.subplot(2, 2, 3)
        plt.plot(self.eval_losses)
        plt.title('Evaluation Loss')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Absolute Difference Loss')

        plt.subplot(2, 2, 4)
        plt.plot(self.eval_rewards)
        plt.title('Evaluation Reward')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Reward')

        plt.tight_layout()
        plt.show()
        