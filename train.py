from arc_challenge_solver.config.config import TRAIN_DIR, EVAL_DIR, PPO_NUM_EPOCHS, PPO_BATCH_SIZE
from arc_challenge_solver.data.arc_dataloader import ARCDataLoader
from arc_challenge_solver.models.ppo_model import PPOModel

def train_ppo_model(task):
    train_pairs = task.get_train_pairs() 
    model = PPOModel(train_pairs) 
    model.train(num_epochs=PPO_NUM_EPOCHS, batch_size=PPO_BATCH_SIZE)
    return model

if __name__ == "__main__":
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    trained_model = train_ppo_model(task)
    # Save the trained model
    trained_model.save("ppo_model.pt")