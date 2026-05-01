import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

    # 训练集的数据预处理：根据 augmentation 配置决定是否使用数据增强
    if config.augmentation == 'random_crop':
        # 使用随机裁剪进行数据增强
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])
    elif config.augmentation == 'flip_h':
        # 使用随机水平翻转进行数据增强
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])
    elif config.augmentation == 'color_jitter':
        # 使用颜色抖动进行数据增强
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])
    else:
        # 不做数据增强，仅做归一化
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
        ])

    # 验证集和测试集使用相同的预处理（不做增强）
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[0.260, 0.257, 0.276]),
    ])

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
