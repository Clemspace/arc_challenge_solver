import os
import torch

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
EVAL_DIR = os.path.join(DATA_DIR, 'evaluation')

# Model parameters
MAX_GRID_SIZE = 30
NUM_COLORS = 10

# PPO hyperparameters
PPO_LEARNING_RATE = 3e-4
PPO_NUM_EPOCHS = 100
PPO_BATCH_SIZE = 64
PPO_CLIP_EPSILON = 0.2
PPO_CRITIC_DISCOUNT = 0.5
PPO_ENTROPY_BETA = 0.01

# Training parameters
RANDOM_SEED = 42

# Evaluation parameters
VISUALIZE_RESULTS = False
SAVE_RESULTS = True
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(BASE_DIR, 'arc_experiment.log')

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Experiment settings
EXPERIMENT_NAME = 'ARC_PPO_vs_Random'