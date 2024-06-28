from config.config import TRAIN_DIR, EVAL_DIR, PPO_NUM_EPOCHS, PPO_BATCH_SIZE
from data.arc_dataloader import ARCDataLoader
from models.ppo_model import PPOModel

def train_ppo_model(task):
    model = PPOModel()
    model.train(task.get_train_pairs())
    return model

if __name__ == "__main__":
    task = ARCDataLoader.load_tasks(TRAIN_DIR, EVAL_DIR)
    trained_model = train_ppo_model(task)
    # Save the trained model
    trained_model.save("ppo_model.pt")