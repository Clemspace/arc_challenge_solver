from arc_challenge_solver.config.config import TRAIN_DIR, EVAL_DIR
from arc_challenge_solver.data.arc_dataloader import ARCDataLoader
from arc_challenge_solver.models.ppo_model import PPOModel
from arc_challenge_solver.models.random_model import RandomModel
from arc_challenge_solver.utils.experiment import ARCExperiment
import arc_challenge_solver.train as train
import arc_challenge_solver.evaluate as evaluate
import arc_challenge_solver.compare_models as compare_models
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