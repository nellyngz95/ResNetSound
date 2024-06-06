import pandas as pd
from pathlib import Path
import random
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio
import yaml
from easydict import EasyDict

metfile= pd.read_csv('/Users/nellygarcia/Documents/InformationRetrivalPhd/Dataset/Processed/metadata.csv')
dataset= Path('/Users/nellygarcia/Documents/InformationRetrivalPhd/Dataset')
# Make a list of classes, converting labels into numbers
labels={}
train_files = metfile[metfile['split'] == 'train']['filename'].tolist()
val_files = metfile[metfile['split'] == 'val']['filename'].tolist()
test_files = metfile[metfile['split'] == 'test']['filename'].tolist()

# Assuming labels are already one-hot encoded in the CSV as lists
train_labels = [eval(label) for label in metfile[metfile['split'] == 'train']['label'].tolist()]
val_labels = [eval(label) for label in metfile[metfile['split'] == 'val']['label'].tolist()]
test_labels = [eval(label) for label in metfile[metfile['split'] == 'test']['label'].tolist()]

with open('/Users/nellygarcia/Documents/InformationRetrivalPhd/config.yaml') as conf:
    cfg = EasyDict(yaml.safe_load(conf))

print(f'Number of train/val/test files are = {len(train_files)}/{len(val_files)}/{len(test_files)}')


#Filename ´+ LAbels 
print("Train Files and Labels:")
for filename, label in zip(train_files, train_labels):
    print(f"Filename: {filename}, Label: {label}")

print("\nValidation Files and Labels:")
for filename, label in zip(val_files, val_labels):
    print(f"Filename: {filename}, Label: {label}")

print("\nTest Files and Labels:")
for filename, label in zip(test_files, test_labels):
    print(f"Filename: {filename}, Label: {label}")



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, filenames, labels, base_path, transforms=None):
        assert len(filenames) == len(labels), f'Inconsistent length of filenames and labels.'

        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.cfg = cfg
        self.base_path = base_path

        # Calculate length of clip this dataset will make
        self.sample_length = int((cfg.clip_length * cfg.sample_rate + cfg.hop_length - 1) // cfg.hop_length)

        # Test with first file
        assert self[0][0].shape[-1] == self.sample_length, f'Check your files, failed to load {filenames[0]}'

        # Show basic info.
        print(f'Dataset will yield log-mel spectrogram {len(self)} data samples in shape [1, {cfg.n_mels}, {self.sample_length}]')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        assert 0 <= index < len(self)

        # Load the log-mel spectrogram
        log_mel_spec = np.load(str(self.base_path / self.filenames[index]))

        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.sample_length - log_mel_spec.shape[-1]
        if pad_size > 0:
            offset = pad_size // 2
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

        # Random crop
        crop_size = log_mel_spec.shape[-1] - self.sample_length
        if crop_size > 0:
            start = np.random.randint(0, crop_size)
            log_mel_spec = log_mel_spec[..., start:start + self.sample_length]

        # Apply augmentations
        if self.transforms is not None:
            log_mel_spec = self.transforms(log_mel_spec)

        return torch.Tensor(log_mel_spec), torch.Tensor(self.labels[index])