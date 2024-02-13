import h5py
import torch
from torch.utils.data import Dataset
from deeprhythm.utils import bpm_to_class

def song_collate(batch):
    # Each element in `batch` is a tuple (song_clips, global_bpm)
    # Where song_clips is a tensor of shape [num_clips, 240, 8, 6]
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels

class SongDataset(Dataset):
    def __init__(self, hdf5_path, group):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            group (str): Group in HDF5 file ('train', 'test', 'validate').
        """
        super(SongDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.group = group
        self.file = h5py.File(hdf5_path, 'r')
        self.group_file = self.file[group]
        self.keys = []
        for key in self.group_file.keys():
            if self.group_file[key].attrs['source'] == 'fma':
                continue
            else:
                self.keys.append(key)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        song_key = self.keys[idx]
        song_data = self.group_file[song_key]
        hcqm = torch.tensor(song_data['hcqm'][:])
        bpm_class = bpm_to_class(int(float(song_data.attrs['bpm'])))
        return hcqm, bpm_class

    def close(self):
        self.file.close()
