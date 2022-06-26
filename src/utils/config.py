from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 128
    learning_rate = 2e-5
    num_train_epochs = 3.0
    warmump_steps = 500
    weight_decay = 0.01