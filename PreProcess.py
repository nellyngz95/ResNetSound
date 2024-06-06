import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import yaml
from easydict import EasyDict
import pandas as pd

root=Path('/Users/nellygarcia/Documents/InformationRetrivalPhd/Dataset')
with open('/Users/nellygarcia/Documents/InformationRetrivalPhd/config.yaml') as conf:
    cfg = EasyDict(yaml.safe_load(conf))

data_files = sorted(root.glob('**/*.wav')
                    )

to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=1024, n_mels=64,
    hop_length=512, f_min=0, f_max=22050)

for file in data_files:
    try:
        # Load waveform
        waveform, sr = torchaudio.load(file)
        
        # Check if the sample rate is correct
        if sr != 44100:
            print(f"Skipping {file} due to incorrect sample rate.")
            continue
        
        # Generate log Mel spectrogram
        log_mel_spec = to_mel_spectrogram(waveform).log()
        
        # Create output directory if it does not exist
        output_dir = root / 'Processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the log Mel spectrogram
        output_file = output_dir / file.with_suffix('.npy').name
        np.save(output_file, log_mel_spec.numpy())
        
        print(f"Processed {file}")
    
    except Exception as e:
        print(f"Failed to process {file}: {e}")