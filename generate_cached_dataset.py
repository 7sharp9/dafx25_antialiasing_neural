"""
Pre-generate and cache teacher model outputs for training.

This script eliminates the data loading bottleneck by:
1. Loading teacher model once on GPU
2. Generating all training samples offline
3. Processing through teacher model in batches
4. Saving cached (x, y, f0, dB) tuples to disk

Usage:
    python generate_cached_dataset.py --config 0 --num_samples 50000 --output cached_train.pt
    python generate_cached_dataset.py --config 0 --num_samples 1000 --output cached_val.pt --validation
"""

import argparse
import sys
import os
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenAmp'))
from Open_Amp.amp_model import AmpModel
from config import get_config


def midi_to_f0(x):
    """Convert MIDI note to frequency in Hz."""
    return 440 * 2 ** ((x - 69) / 12)


def f0_to_midi(x):
    """Convert frequency in Hz to MIDI note."""
    return 12 * torch.log2(x / 440) + 69


def generate_sine_tone(conf, randomise=True, index=0):
    """
    Generate a single sine tone with random or fixed parameters.

    Args:
        conf: Configuration dict with train_data or val_data params
        randomise: If True, random f0/gain. If False, use index
        index: Sample index (used when randomise=False)

    Returns:
        x: Input signal [time_samples, 1]
        f0: Fundamental frequency (scalar tensor)
        dB: Gain in dB (scalar tensor)
    """
    sample_rate = conf['sample_rate']
    dur = conf['dur']
    midi_min = conf['midi_min']
    midi_max = conf.get('midi_max', 127)
    dB_min = conf['dB_min']
    dB_max = conf['dB_max']
    linear_f0_sample = conf.get('linear_f0_sample', False)

    time = torch.arange(0, int(dur * sample_rate), dtype=torch.double) / sample_rate

    if randomise:
        if linear_f0_sample:
            f0_min = midi_to_f0(torch.tensor(midi_min))
            f0_max = midi_to_f0(torch.tensor(midi_max))
            f0 = f0_min + (f0_max - f0_min) * torch.rand(1)
        else:
            midi = midi_min + (midi_max - midi_min) * torch.rand(1)
            f0 = midi_to_f0(midi)

        gain = (10 ** (dB_max / 20)) * torch.rand(1)
        dB = 20 * torch.log10(gain)
        phi = 2 * torch.pi * torch.rand(1)
    else:
        # Fixed parameters based on index
        midi = torch.tensor([midi_min + index])
        f0 = midi_to_f0(midi)
        dB = torch.tensor([dB_min])
        gain = 10 ** (dB / 20)
        phi = torch.zeros(1)

    x = gain * torch.sin(2 * torch.pi * f0 * time + phi).unsqueeze(-1)

    return x, f0, dB


def process_through_teacher(model, x_batch, model_class='WaveNet'):
    """
    Process batch of inputs through teacher model.

    Args:
        model: Teacher model (already on correct device)
        x_batch: Batch of inputs [batch, time, channels]
        model_class: Model architecture type

    Returns:
        y_batch: Teacher outputs [batch, time, channels]
    """
    if model_class == 'SimpleRNN':
        # RNN requires frame-by-frame processing
        batch_size = x_batch.shape[0]
        frame_size = 4410
        num_frames = int(np.floor(x_batch.shape[1] / frame_size))

        outputs = []
        for b in range(batch_size):
            model.model.reset_state()
            y_frames = []
            for n in range(num_frames):
                start = frame_size * n
                end = frame_size * (n + 1)
                x_frame = x_batch[b:b+1, start:end, :]
                y_frame = model(x_frame)
                y_frames.append(y_frame)
            y = torch.cat(y_frames, dim=1)
            outputs.append(y)
        return torch.cat(outputs, dim=0)
    else:
        # WaveNet can process entire batch at once
        return model(x_batch)


def generate_cached_dataset(args):
    """Generate and cache teacher model outputs."""

    # Load config
    conf = get_config(args.config)

    # Choose train or validation data config
    if args.validation:
        data_conf = conf['val_data'].copy()
        data_conf['sample_rate'] = conf['sample_rate']
        randomise = data_conf.get('randomise', False)
        num_samples = data_conf.get('num_tones', 88)
        print(f"Generating VALIDATION dataset with {num_samples} samples")
    else:
        data_conf = conf['train_data'].copy()
        data_conf['sample_rate'] = conf['sample_rate']
        randomise = True
        num_samples = args.num_samples
        print(f"Generating TRAINING dataset with {num_samples} samples")

    # Add missing fields to data_conf
    if 'midi_max' not in data_conf:
        data_conf['midi_max'] = 127
    if 'dB_max' not in data_conf:
        data_conf['dB_max'] = 0

    print(f"Configuration: {conf['model_name']}")
    print(f"Teacher model: {conf['model_json']}")
    print(f"Sample rate: {conf['sample_rate']} Hz")
    print(f"Duration: {data_conf['dur']} seconds")
    print(f"Frequency range: MIDI {data_conf['midi_min']} - {data_conf['midi_max']}")
    print(f"Amplitude range: {data_conf['dB_min']} - {data_conf['dB_max']} dB")

    # Load teacher model
    print("\nLoading teacher model...")
    model = AmpModel(conf['model_json'], conf['model_name'])
    model.double()
    model.eval()
    model.requires_grad_(False)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Teacher model loaded on {device}")
    print(model)

    # Generate dataset
    print(f"\nGenerating {num_samples} samples...")
    dataset = []

    # Process in batches for efficiency
    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Determine batch size (last batch may be smaller)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # Generate batch of sine tones
            x_batch = []
            f0_batch = []
            dB_batch = []

            for i in range(current_batch_size):
                sample_idx = start_idx + i
                x, f0, dB = generate_sine_tone(data_conf, randomise=randomise, index=sample_idx)
                x_batch.append(x)
                f0_batch.append(f0)
                dB_batch.append(dB)

            x_batch = torch.stack(x_batch).to(device)  # [batch, time, 1]

            # Process through teacher model
            y_batch = process_through_teacher(model, x_batch, model.model_class)

            # Move back to CPU and store
            x_batch = x_batch.cpu()
            y_batch = y_batch.cpu()

            for i in range(current_batch_size):
                dataset.append((
                    x_batch[i],      # Input sine tone
                    y_batch[i],      # Teacher output
                    f0_batch[i],     # Fundamental frequency
                    dB_batch[i]      # Amplitude in dB
                ))

    # Save to disk
    print(f"\nSaving cached dataset to {args.output}...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    torch.save(dataset, args.output)

    # Report statistics
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✓ Saved {len(dataset)} samples ({file_size_mb:.1f} MB)")
    print(f"  Sample shape: x={dataset[0][0].shape}, y={dataset[0][1].shape}")

    # Quick validation check
    print("\nValidation check:")
    print(f"  f0 range: {torch.stack([f0 for _, _, f0, _ in dataset]).min():.1f} - {torch.stack([f0 for _, _, f0, _ in dataset]).max():.1f} Hz")
    print(f"  dB range: {torch.stack([dB for _, _, _, dB in dataset]).min():.1f} - {torch.stack([dB for _, _, _, dB in dataset]).max():.1f} dB")
    print(f"  x energy range: {torch.stack([x.pow(2).mean() for x, _, _, _ in dataset]).min():.2e} - {torch.stack([x.pow(2).mean() for x, _, _, _ in dataset]).max():.2e}")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate cached teacher outputs")
    parser.add_argument("--config", type=int, default=0, help="Model config index")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of training samples to generate")
    parser.add_argument("--output", type=str, default="cached_train.pt", help="Output file path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for teacher inference")
    parser.add_argument("--validation", action="store_true", help="Generate validation dataset instead of training")

    args = parser.parse_args()

    print("=" * 70)
    print("Teacher Model Output Pre-caching Script")
    print("=" * 70)

    generate_cached_dataset(args)

    print("\n" + "=" * 70)
    print("Done! Use this cached dataset in training with --use_cached_data flag")
    print("=" * 70)


if __name__ == '__main__':
    main()
