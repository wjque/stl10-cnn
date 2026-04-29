from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    name: str
    seed: int = 42
    num_epochs: int = 500
    batch_size: int = 64
    num_workers: int = 4

    augmentation: str = 'none'
    use_residual: bool = False
    depth: str = 'shallow'
    activation: str = 'relu'
    pooling: str = 'max'
    use_bn: bool = False

    dropout: float = 0.5
    optimizer_name: str = 'sgd'
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9

    mixup_alpha: float = 1.0
    patience: int = 20

    model_save_path: str = ''
    log_save_path: str = ''

    def __post_init__(self):
        if not self.model_save_path:
            self.model_save_path = f'outputs/models/{self.name}.pth'
        if not self.log_save_path:
            self.log_save_path = f'outputs/logs/{self.name}.json'
