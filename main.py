import tyro

from joermungandr.config import ModelConfig, TrainConfig
from joermungandr.encoder import train_loop

if __name__ == "__main__":
    model, train = tyro.cli(tuple[ModelConfig, TrainConfig])
    train_loop(model, train)
