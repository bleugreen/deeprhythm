import h5py
import torch
from torch.utils.data import Dataset
from deeprhythm.utils import bpm_to_class
class ClipDataset(Dataset):
    def __init__(self, hdf5_file, group, use_float=False):
        """
        :param hdf5_file: Path to the HDF5 file.
        :param group: Group in the HDF5 file to use ('train', 'test', 'validate').
        """
        self.use_float = use_float
        self.hdf5_file = hdf5_file
        self.group = group
        self.index_map = []
        self.file_ref = h5py.File(self.hdf5_file, 'r')
        group_data = self.file_ref[group]
        for song_key in group_data.keys():
            song_data = group_data[song_key]
            if song_data.attrs['source'] == 'fma':
                continue
            num_clips = song_data['hcqm'].shape[0]
            if num_clips > 5:
                clip_start = 1
                clip_range = num_clips-2
            else:
                clip_start, clip_range = 0, num_clips

            for clip_index in range(clip_start, clip_range):
                self.index_map.append((song_key, clip_index))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_key, clip_index = self.index_map[idx]
        song_data = self.file_ref[self.group][song_key]
        hcqm = song_data['hcqm'][clip_index]
        bpm = torch.tensor(float(song_data.attrs['bpm']), dtype=torch.float32)
        hcqm_tensor = torch.tensor(hcqm, dtype=torch.float).permute(2, 0, 1)
        if self.use_float:
            return hcqm_tensor, bpm
        label_class_index = bpm_to_class(int(bpm))  # Convert BPM to class index
        return hcqm_tensor, label_class_index
