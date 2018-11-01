from architectures.DQL import DQLValue

parameters = {
    "epsilon_decay": 0.001,
    "min_epsilon": 0.1,
    "enviroment": "CarRacing-v0",
    "model": DQLValue(),
    "max_episodes": 5000,
    "test_step": 10,
    "checkpoint_path": "./models",
    "tensorboard_path": "./logs",
    "save_step": 50
}