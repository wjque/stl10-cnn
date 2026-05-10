import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import Config
from model.cnn import CNNFactory
from utils.dataloader import create_dataloaders
from utils.metrics import compute_metrics


def load_model_from_log(log_path, device):
    with open(log_path, 'r') as f:
        log_data = json.load(f)

    config_dict = log_data['config']
    model_path = config_dict.get('model_save_path', log_path.replace('logs', 'models').replace('.json', '.pth'))

    model = CNNFactory(
        num_classes=10,
        depth=config_dict.get('depth', 'shallow'),
        pooling=config_dict.get('pooling', 'max'),
        use_bn=config_dict.get('use_bn', False),
        dropout=config_dict.get('dropout', 0.0),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, log_data


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    all_scores = []
    all_targets = []

    for inputs, targets in tqdm(loader, desc='Inference'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_scores.append(outputs.softmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())

    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    all_preds = all_scores.argmax(axis=1)
    return compute_metrics(all_targets, all_preds, all_scores)


def evaluate_experiment(experiment_name, split='test'):
    log_path = f'outputs/logs/{experiment_name}.json'
    if not os.path.exists(log_path):
        raise FileNotFoundError(f'Log not found: {log_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, log_data = load_model_from_log(log_path, device)
    config = Config(**log_data['config'])
    _, val_loader, test_loader, _ = create_dataloaders(config)

    if split == 'val':
        return infer(model, val_loader, device)
    if split == 'test':
        return infer(model, test_loader, device)
    raise ValueError(f'Unknown split: {split}')
