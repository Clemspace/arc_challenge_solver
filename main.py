from config.config import TRAIN_DIR, EVAL_DIR
from data.arc_dataloader import ARCDataLoader
from models.ppo_model import PPOModel
from models.random_model import RandomModel
from utils.experiment import ARCExperiment
import train
import evaluate
import compare_models

def run_full_experiment():
    print("Loading ARC tasks...")
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    
    print("Training PPO model...")
    ppo_model = train.train_ppo_model(task)
    ppo_model.save("ppo_model.pt")
    
    print("Evaluating models...")
    models = {
        "PPO": ppo_model,
        "Random": RandomModel()
    }
    results = compare_models.compare_models(models, task)
    
    print("Plotting comparison...")
    compare_models.plot_comparison(results)
    
    print("Experiment completed!")

if __name__ == "__main__":
    run_full_experiment()