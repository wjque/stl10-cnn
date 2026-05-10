import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MEAN = [0.447, 0.440, 0.407]
STD = [0.260, 0.257, 0.276]


def build_train_transform(config):
    transform_steps = []
    augmentations = getattr(config, 'augmentations', None)
    if augmentations is None:
        augmentation = getattr(config, 'augmentation', 'none')
        augmentations = [] if augmentation == 'none' or augmentation is None else [augmentation]

    for augmentation in augmentations:
        if augmentation == 'random_crop':
            transform_steps.append(transforms.RandomResizedCrop(config.input_size, scale=(0.8, 1.0)))
        elif augmentation == 'flip_h':
            transform_steps.append(transforms.RandomHorizontalFlip())
        else:
            raise ValueError(f'Unknown augmentation: {augmentation}')

    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transforms.Compose(transform_steps)


def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def create_dataloaders(config, data_dir='STL10'):
    """
    Args:
        config: Config dataclass with batch_size, num_workers, augmentation.
        data_dir: Root directory containing train/val/test subfolders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform()

    # 使用 ImageFolder 从目录结构自动加载图像和标签
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes
