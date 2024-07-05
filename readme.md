# ARC Challenge Solver with Neural Architecture Search

This project implements a Neural Architecture Search (NAS) approach to solve the Abstraction and Reasoning Corpus (ARC) Challenge using Proximal Policy Optimization (PPO) and NNI (Neural Network Intelligence).

## Project Overview

The ARC Challenge presents a series of tasks that test an AI system's ability to learn and apply abstract reasoning. This solver uses a PPO model with a customizable neural architecture to tackle these tasks. The architecture is optimized using NNI to search for the most effective configuration.

## Key Features

- Implementation of PPO for ARC task solving
- Neural Architecture Search using NNI
- Flexible model architecture with configurable hyperparameters
- Support for variable-sized input and output grids (up to 30x30)
- Reward function considering both grid content and size accuracy

## Project Structure
arc_challenge_solver/
│
├── analysis.ipynb
├── compare_models.py
├── evaluate.py
├── LICENCE
├── main.py
├── readme.md
├── requirements.txt
├── test.py
├── train.py
│
├── config/
│   └── config.py
│
├── data/
│   ├── arc_dataloader.py
│   ├── arc_task.py
│   ├── evaluation/
│   └── training/
│
├── models/
│   ├── base_model.py
│   ├── ppo_model.py
│   └── random_model.py
│
├── nas/
│   ├── analyze_nas_results.py
│   ├── config.yml
│   └── nas_trial.py
│
└── utils/
├── checkpoint.py
├── experiment.py
├── loss_functions.py
├── metrics.py
└── visualizer.py

## Setup

1. Clone the repository:
git clone https://github.com/clemspace/arc-challenge-solver.git
cd arc-challenge-solver

1. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

1. Install the required packages:
pip install -r requirements.txt

1. Download the ARC dataset and place it in the appropriate directories under `data/evaluation/` and `data/training/`.

## Usage

To run the Neural Architecture Search:
nnictl create --config nas/config.yml

This will start the NAS process using the configuration specified in `config.yml`. The search will explore different neural architectures and hyperparameters to optimize the PPO model's performance on ARC tasks.

To run other components:

- Train the PPO model: `python train.py`
- Evaluate models: `python evaluate.py`
- Compare models: `python compare_models.py`
- Run the full pipeline: `python main.py`

## Model Architecture

The PPO model uses a convolutional neural network with the following key components:

- Variable number of convolutional layers
- Skip connections (optional)
- Dropout for regularization
- Flexible activation functions
- Size prediction for variable output sizes

## Neural Architecture Search

The NAS process optimizes the following hyperparameters:

- Learning rate
- Batch size
- Number of layers
- Hidden dimension size
- Dropout rate
- Optimizer selection
- Activation function
- Use of skip connections
- Weight decay

## Results

Current best reward: 0.5456464971814837 (starting from 0.0614475453538554)

The significant improvement in reward suggests that the NAS process is effectively optimizing the model architecture and hyperparameters for the ARC tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC)
- [OpenAI Spinning Up](https://spinningup.openai.com/) for PPO implementation guidance
- [NNI (Neural Network Intelligence)](https://github.com/microsoft/nni) for Neural Architecture Search