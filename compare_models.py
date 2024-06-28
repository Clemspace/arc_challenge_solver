from config.config import EVAL_DIR, TRAIN_DIR
from data.arc_dataloader import ARCDataLoader
from models.ppo_model import PPOModel
from models.random_model import RandomModel
from utils.experiment import ARCExperiment
import matplotlib.pyplot as plt

def compare_models(models, task):
    results = {}
    for name, model in models.items():
        experiment = ARCExperiment(task, model)
        accuracy, avg_loss = experiment.evaluate()
        results[name] = {"accuracy": accuracy, "avg_loss": avg_loss}
    return results

def plot_comparison(results):
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]
    losses = [results[name]["avg_loss"] for name in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(names, accuracies)
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylabel("Accuracy")
    
    ax2.bar(names, losses)
    ax2.set_title("Model Average Loss Comparison")
    ax2.set_ylabel("Average Loss")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    
    models = {
        "PPO": PPOModel.load("ppo_model.pt"),
        "Random": RandomModel()
    }
    
    results = compare_models(models, task)
    plot_comparison(results)