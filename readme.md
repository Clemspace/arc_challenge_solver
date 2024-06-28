# ARC Challenge Solver

This project implements and compares different approaches to solve the Abstraction and Reasoning Corpus (ARC) Challenge.

## Project Structure

```
arc_challenge/
│
├── data/
│   ├── arc_task.py
│   └── arc_dataloader.py
│
├── models/
│   ├── base_model.py
│   ├── random_model.py
│   └── ppo_model.py
│
├── utils/
│   ├── loss_functions.py
│   ├── visualizer.py
│   └── experiment.py
│
├── config/
│   └── config.py
│
├── train.py
├── evaluate.py
├── compare_models.py
└── main.py
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/arc-challenge-solver.git
   cd arc-challenge-solver
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the ARC dataset and place it in the appropriate directories as specified in `config/config.py`.

## Usage

To run the full experiment pipeline:

```
python main.py
```

This will train the PPO model, evaluate both the PPO and Random models, and compare their performance.

To run individual components:

- Train the PPO model: `python train.py`
- Evaluate models: `python evaluate.py`
- Compare models: `python compare_models.py`

## Models

- **Random Model**: Generates random outputs as a baseline.
- **PPO Model**: Implements basic Proximal Policy Optimization to learn and solve ARC tasks.

## Configuration

Adjust hyperparameters and settings in `config/config.py`.

## Results

Results will be displayed in the console and saved in the `results` directory (if specified in the configuration).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC)
- [OpenAI Spinning Up](https://spinningup.openai.com/) for PPO implementation guidance

```