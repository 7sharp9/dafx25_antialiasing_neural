from torch.utils.data import Dataset
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenAmp'))
from Open_Amp.amp_model import AmpModel
import numpy as np
from spectral import cheb_fft

import matplotlib as mpl
# mpl.use("macosx")

def midi_to_f0(x):
    return 440 * 2 ** ((x - 69) / 12)

def f0_to_midi(x):
    return 12 * torch.log2(x/440) + 69

class SineToneDataset(Dataset):

    def __init__(self, device,
                 sample_rate = 44100, dur=1.1,
                 midi_min=21, midi_max=127,
                 dB_min=-60, dB_max=0,
                 num_tones=1024,
                 randomise=True,
                 linear_f0_sample=False):
        self.model = AmpModel(device, '')
        self.model.double()
        self.model.requires_grad_(False)
        self.midi_min = midi_min
        self.midi_max = midi_max
        self.dB_min = dB_min
        self.dB_max = dB_max
        self.sample_rate = sample_rate
        self.time = torch.arange(0, int(dur * sample_rate), dtype=torch.double) / sample_rate
        self.randomise = randomise
        self.len = num_tones
        self.linear_f0_sample=linear_f0_sample
        print(self.model)

    def __getitem__(self, index):

        if self.randomise:
            if self.linear_f0_sample:
                f0_min = midi_to_f0(self.midi_min)
                f0_max = midi_to_f0(self.midi_max)
                f0 = f0_min + (f0_max - f0_min) * torch.rand(1)
                midi = f0_to_midi(f0)
            else:
                midi = self.midi_min + (self.midi_max - self.midi_min) * torch.rand(1)
            gain = (10 ** (self.dB_max/20)) * torch.rand(1)
            dB = 20 * torch.log10(gain)
            phi = 2 * torch.pi * torch.rand(1)
        else:
            midi = torch.Tensor([self.midi_min + index])
            dB = self.dB_min
            gain = 10 ** (dB/20)
            phi = torch.zeros(1)


        f0 = midi_to_f0(midi)
        x = gain * torch.sin(2 * torch.pi * f0 * self.time + phi).unsqueeze(-1)

        if self.model.model_class == 'SimpleRNN':
            self.model.model.reset_state()

            frame_size = 4410
            num_frames = int(np.floor(x.shape[0] / frame_size))
            y = []
            for n in range(num_frames):
                start = frame_size * n
                end = frame_size * (n + 1)
                x_frame = x[start:end, :]
                y_frame = self.model(x_frame)
                y.append(y_frame)
            y = torch.cat(y, dim=0)
        else:
            y = self.model(x).squeeze(-1)


        return x, y, f0, dB

    def __len__(self):
        return self.len


class SequenceDataset(Dataset):
    def __init__(self, input, sequence_length, target=None, device=None):

        if device is not None:
            self.model = AmpModel(device, '')
            self.model.double()
            self.model.requires_grad_(False)

        if sequence_length is None:
            self._sequence_length = input.shape[1]
        else:
            self._sequence_length = sequence_length

        self.input = input
        self.input_sequence = self.wrap_to_sequences(self.input, self._sequence_length)
        self._len = self.input_sequence.shape[0]

        if target is None:
            print('Processing data through model:', device)
            self.target_sequence = self.model(self.input_sequence)
        else:
            self.target_sequence = self.wrap_to_sequences(target, self._sequence_length)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.input_sequence[index, :, :], self.target_sequence[index, :, :]

    # wraps data from  [channels, samples] -> [sequences, samples, channels]
    def wrap_to_sequences(self, data, sequence_length):
        num_sequences = int(np.floor(data.shape[1] / sequence_length))
        truncated_data = data[:, 0:(num_sequences * sequence_length)]
        wrapped_data = truncated_data.transpose(0, 1).reshape((num_sequences, sequence_length, data.shape[0]))
        return wrapped_data


class CachedSineToneDataset(Dataset):
    """
    Dataset that loads pre-cached teacher outputs from disk.

    This eliminates the data loading bottleneck by loading pre-processed
    (x, y, f0, dB) tuples instead of running teacher inference on-the-fly.

    Args:
        cache_path: Path to cached dataset file (.pt)
    """

    def __init__(self, cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        self.data = torch.load(cache_path)
        print(f"Loaded {len(self.data)} cached samples")

        # Validate data format
        if len(self.data) > 0:
            sample = self.data[0]
            assert len(sample) == 4, "Cached data must be (x, y, f0, dB) tuples"
            print(f"  Sample shapes: x={sample[0].shape}, y={sample[1].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y, f0, dB = self.data[index]
        return x, y, f0, dB


class SineToneLoadingDataset(SequenceDataset):

    def __init__(self,
                 input,
                 target,
                 sequence_length=2):
        super().__init__(input=input, target=target, sequence_length=sequence_length)

        f0s = []
        midis = []
        for i in range(self.input_sequence.shape[0]):
            x = self.input_sequence[i, -44100:, 0]
            in_fft = cheb_fft(x)
            closest_f0 = torch.argmax(torch.abs(in_fft))
            closest_midi = torch.round(f0_to_midi(closest_f0))
            f0 = midi_to_f0(closest_midi)
            f0s.append(f0)
            midis.append(closest_midi)


        self.f0 = torch.stack(f0s)
        self.midi = torch.stack(midis)

        # ensure midi value is integer sequence
        assert(all(torch.diff(self.midi) == torch.ones_like(self.midi)[:-1]))


    def __getitem__(self, index):
        return self.input_sequence[index, :, :], \
            self.target_sequence[index, :, :], \
            self.f0[index], \
            self.midi[index]






