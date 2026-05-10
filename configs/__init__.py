from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    name: str
    stage: str = ''
    seed: int = 42
    num_epochs: int = 500
    batch_size: int = 64
    num_workers: int = 4
    input_size: int = 96

    augmentations: list[str] = field(default_factory=list)
    augmentation: Optional[str] = None
    depth: str = 'shallow'
    activation: str = 'relu'
    pooling: str = 'max'
    use_bn: bool = False

    dropout: float = 0.0
    optimizer_name: str = 'sgd'
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9

    scheduler_name: str = 'none'
    scheduler_t_max: Optional[int] = None
    use_early_stopping: bool = False

    patience: int = 20
    notes: str = ''

    model_save_path: str = ''
    log_save_path: str = ''

    def __post_init__(self):
        self.augmentations = self._normalize_augmentations(self.augmentations, self.augmentation)
        self.augmentation = self.augmentations[0] if len(self.augmentations) == 1 else None

        if not self.model_save_path:
            self.model_save_path = f'outputs/models/{self.name}.pth'
        if not self.log_save_path:
            self.log_save_path = f'outputs/logs/{self.name}.json'

    @staticmethod
    def _normalize_augmentations(augmentations, augmentation):
        if augmentations is None:
            augmentations = []
        if isinstance(augmentations, str):
            augmentations = [] if augmentations == 'none' else [augmentations]
        if augmentation and augmentation != 'none':
            if augmentation not in augmentations:
                augmentations.append(augmentation)
        return list(augmentations)

    def clone(self, **updates):
        data = self.__dict__.copy()
        data.update(updates)
        return Config(**data)
