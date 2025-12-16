# Cached Dataset Usage Guide

This guide explains how to use pre-cached teacher outputs to eliminate the data loading bottleneck.

## Problem

The original training pipeline runs teacher model inference on CPU in worker threads during training, creating a 5-10x slowdown. With 10 workers each running sequential model inference, the GPU sits idle while waiting for data.

## Solution

Pre-generate all teacher outputs once using the GPU, save to disk, and load from cache during training.

**Expected speedup**: 5-10x faster epoch time

---

## Step 1: Generate Cached Datasets

### Training Dataset

Generate a large training dataset (e.g., 50,000 samples):

```bash
python generate_cached_dataset.py \
    --config 0 \
    --num_samples 50000 \
    --output cached_train.pt \
    --batch_size 64
```

### Validation Dataset

Generate validation dataset using config's validation parameters:

```bash
python generate_cached_dataset.py \
    --config 0 \
    --validation \
    --output cached_val.pt \
    --batch_size 64
```

### Options

- `--config N`: Which model config to use (0-7, see config.py)
- `--num_samples N`: Number of training samples (only for training, validation uses config)
- `--output PATH`: Where to save the cached dataset (.pt file)
- `--batch_size N`: Batch size for teacher inference (larger = faster, needs more GPU memory)
- `--validation`: Generate validation dataset instead of training

---

## Step 2: Train with Cached Data

Use the `--use_cached_data` flag when training:

```bash
python train.py \
    --config 0 \
    --wandb \
    --use_cached_data \
    --cached_train_path cached_train.pt \
    --cached_val_path cached_val.pt
```

### Options

- `--use_cached_data`: Enable cached dataset mode
- `--cached_train_path PATH`: Path to cached training data (default: cached_train.pt)
- `--cached_val_path PATH`: Path to cached validation data (default: cached_val.pt)

---

## Example Workflow

```bash
# 1. Generate training dataset (50k samples)
python generate_cached_dataset.py \
    --config 0 \
    --num_samples 50000 \
    --output cached_train.pt

# 2. Generate validation dataset
python generate_cached_dataset.py \
    --config 0 \
    --validation \
    --output cached_val.pt

# 3. Train with cached data
python train.py \
    --config 0 \
    --wandb \
    --use_cached_data
```

---

## Storage Requirements

Approximate file sizes for cached datasets:

| Samples | Duration | File Size (approx) |
|---------|----------|-------------------|
| 1,600   | 1.2s     | ~80 MB            |
| 10,000  | 1.2s     | ~500 MB           |
| 50,000  | 1.2s     | ~2.5 GB           |
| 100,000 | 1.2s     | ~5 GB             |

Formula: ~50 KB per sample (1.2s @ 44.1 kHz, float64)

---

## Verification

After generating cached datasets, verify them:

```python
import torch

# Load and inspect
data = torch.load('cached_train.pt')
print(f"Samples: {len(data)}")

x, y, f0, dB = data[0]
print(f"x shape: {x.shape}")  # Should be [time_samples, 1]
print(f"y shape: {y.shape}")  # Should be [time_samples, 1]
print(f"f0 range: {torch.stack([f for _, _, f, _ in data]).min():.1f} - {torch.stack([f for _, _, f, _ in data]).max():.1f} Hz")
print(f"dB range: {torch.stack([d for _, _, _, d in data]).min():.1f} - {torch.stack([d for _, _, _, d in data]).max():.1f} dB")
```

---

## Comparison: On-the-fly vs Cached

### On-the-fly (Original)
- Teacher inference on CPU in 10 worker threads
- GPU sits idle during data loading
- `batch_load_time` metric shows bottleneck
- **Epoch time**: ~5-10 minutes per epoch (typical)

### Cached (New)
- Teacher inference once on GPU (offline)
- Workers only load from disk (fast)
- GPU continuously busy with student training
- **Epoch time**: ~0.5-1 minute per epoch (estimated)

---

## Notes

1. **Regenerate cache if you change**:
   - Model config (different teacher)
   - Sample rate
   - Duration
   - Frequency/amplitude ranges

2. **Cache is deterministic for validation** (randomise=False), so validation metrics are comparable across runs.

3. **Cache is randomized for training** (different samples each time you regenerate).

4. **GPU usage**: Cache generation uses GPU efficiently (batched inference). Original training does not (CPU workers).

5. **Colab**: Store cached datasets in Google Drive to avoid regenerating each session:
   ```bash
   python generate_cached_dataset.py \
       --output /content/drive/MyDrive/AA_Neural/cached_train.pt
   ```
