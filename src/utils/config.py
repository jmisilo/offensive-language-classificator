from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int  = 6
    warmump_steps: int = 500
    weight_decay: float = 0.01
    max_norm: float = 2.0
    num_workers: int = 2
    pin_memory: bool = True