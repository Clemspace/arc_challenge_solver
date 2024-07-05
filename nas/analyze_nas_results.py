import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from arc_challenge_solver.utils.checkpoint import save_nas_results, load_nas_result

def analyze_nas_run(experiment_dir: str):
    results = []
    for trial_id in os.listdir(experiment_dir):
        trial_dir = os.path.join(experiment_dir, trial_id)
        model_state_path = os.path.join(trial_dir, 'model_state.pth')
        if os.path.exists(model_state_path):
            model_state = load_nas_result(model_state_path)
            results.append((model_state['score'], model_state))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    best_model = results[0][1]
    median_model = results[len(results)//2][1]
    worst_model = results[-1][1]
    
    save_nas_results(best_model, median_model, worst_model, 'nas_analysis_results')

    print(f"Best score: {best_model['score']}")
    print(f"Median score: {median_model['score']}")
    print(f"Worst score: {worst_model['score']}")

if __name__ == "__main__":
    analyze_nas_run(r'arc_challenge_solver/nas/nas_results')