import tyro

from config import ModelConfig, TrainConfig
from train import train_loop

if __name__ == "__main__":
    model, train = tyro.cli(tuple[ModelConfig, TrainConfig])
    train_loop(model, train)
