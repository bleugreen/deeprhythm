import h5py
from torch.utils.data import Dataset,Subset
import torch
from torch.utils.data.dataset import random_split
import json

def bpm_to_class(bpm, min_bpm=30, max_bpm=286, num_classes=256):
    """Map a BPM value to a class index."""
    # Linearly map BPM values to class indices
    class_width = (max_bpm - min_bpm) / num_classes
    class_index = int((bpm - min_bpm) // class_width)
    return max(0, min(num_classes - 1, class_index))

def class_to_bpm(class_index, min_bpm=30, max_bpm=286, num_classes=256):
    """Map a class index back to a BPM value (to the center of the class interval)."""
    class_width = (max_bpm - min_bpm) / num_classes
    bpm = min_bpm + class_width * (class_index + 0.5)
    return bpm

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.file_path = hdf5_file
        self.transform = transform
        with h5py.File(self.file_path, 'r') as file:
            self.items = []
            for group_name in file.keys():
                group = file[group_name]
                for item_name in group.keys():
                    self.items.append((group_name, item_name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        group_name, item_name = self.items[idx]
        with h5py.File(self.file_path, 'r') as file:
            item = file[group_name][item_name]
            data = torch.tensor(item['hcqm'][:], dtype=torch.float)
            bpm = torch.tensor([item.attrs['bpm']], dtype=torch.int)
        label_class_index = bpm_to_class(bpm)  # Convert BPM to class index
        data = data.permute(2, 0, 1)
        return data, label_class_index

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
    # Extract indices from the subsets
    splits = {
        'train_indices': train_dataset.indices,
        'test_indices': test_dataset.indices,
        'validate_indices': validate_dataset.indices
    }
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(splits, f)


def load_split_datasets(dataset, filename="dataset_splits.json"):
    # Load the saved indices
    with open(filename, 'r') as f:
        splits = json.load(f)

    # Recreate the subsets using the loaded indices
    train_dataset = Subset(dataset, splits['train_indices'])
    test_dataset = Subset(dataset, splits['test_indices'])
    validate_dataset = Subset(dataset, splits['validate_indices'])

    return train_dataset, test_dataset, validate_dataset