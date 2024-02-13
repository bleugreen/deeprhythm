from torch.utils.data import Subset
from torch.utils.data.dataset import random_split
import json

def split_dataset(dataset, train_ratio, test_ratio, validate_ratio):
    total_ratio = train_ratio + test_ratio + validate_ratio
    assert abs(total_ratio - 1) < 1e-6, "Ratios must sum to 1"

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    validate_size = dataset_size - train_size - test_size

    train_dataset, test_dataset, validate_dataset = random_split(dataset, [train_size, test_size, validate_size])
    return train_dataset, test_dataset, validate_dataset

def save_split_indices(train_dataset, test_dataset, validate_dataset, filename="dataset_splits.json"):
    splits = {
        'train_indices': train_dataset.indices,
        'test_indices': test_dataset.indices,
        'validate_indices': validate_dataset.indices
    }
    with open(filename, 'w') as f:
        json.dump(splits, f)


def load_split_datasets(dataset, filename="dataset_splits.json"):
    with open(filename, 'r') as f:
        splits = json.load(f)

    train_dataset = Subset(dataset, splits['train_indices'])
    test_dataset = Subset(dataset, splits['test_indices'])
    validate_dataset = Subset(dataset, splits['validate_indices'])

    return train_dataset, test_dataset, validate_dataset