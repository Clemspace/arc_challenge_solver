import numpy as np

def task_difficulty(task):
    input_grid = np.array(task['input'])
    output_grid = np.array(task['output'])
    
    # Calculate difficulty based on grid size and number of unique colors
    size_difficulty = input_grid.size + output_grid.size
    color_difficulty = len(np.unique(input_grid)) + len(np.unique(output_grid))
    
    return size_difficulty * color_difficulty

def sort_tasks_by_difficulty(tasks):
    return sorted(tasks, key=task_difficulty)

def get_curriculum(tasks, num_stages=5):
    sorted_tasks = sort_tasks_by_difficulty(tasks)
    tasks_per_stage = len(sorted_tasks) // num_stages
    
    curriculum = []
    for i in range(num_stages):
        start_idx = i * tasks_per_stage
        end_idx = (i + 1) * tasks_per_stage if i < num_stages - 1 else len(sorted_tasks)
        curriculum.append(sorted_tasks[start_idx:end_idx])
    
    return curriculum