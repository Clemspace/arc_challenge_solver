from config.config import EVAL_DIR, TRAIN_DIR
from data.arc_dataloader import ARCDataLoader
from models.ppo_model import PPOModel
from models.random_model import RandomModel
from utils.experiment import ARCExperiment

def evaluate_model(model, task):
    experiment = ARCExperiment(task, model)
    accuracy, avg_loss = experiment.evaluate()
    return accuracy, avg_loss

if __name__ == "__main__":
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    
    # Evaluate PPO model
    ppo_model = PPOModel.load("ppo_model.pt")
    ppo_accuracy, ppo_loss = evaluate_model(ppo_model, task)
    print(f"PPO Model - Accuracy: {ppo_accuracy:.2f}, Avg Loss: {ppo_loss:.2f}")
    
    # Evaluate Random model
    random_model = RandomModel()
    random_accuracy, random_loss = evaluate_model(random_model, task)
    print(f"Random Model - Accuracy: {random_accuracy:.2f}, Avg Loss: {random_loss:.2f}")