import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(config, data_dir='STL10'):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    if config.augmentation == 'random_crop':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
    ])

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
