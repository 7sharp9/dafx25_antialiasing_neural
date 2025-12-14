# Google Colab Training Notebook Design

**Date:** 2025-12-14
**Purpose:** Enable GPU-accelerated training on Google Colab to reduce epoch time from 32 minutes (local CPU) to ~1-2 minutes (Colab T4 GPU)

## Design Approach

**Hybrid architecture:** Git clone for code, Google Drive for persistent data.

**Key decisions:**
- Minimal wrapper notebook (keeps train.py unchanged)
- wandb for remote monitoring
- Focus on idx=3 (JCM800 NAM) with flexibility for other models
- Drive persistence for checkpoints and weights

## Notebook Structure

### Cell Organization

1. **Setup cell:** Mount Google Drive to `/content/drive`
2. **Environment cell:** Install dependencies (pip-based, converted from env.yaml)
3. **Clone cell:** Git clone repo to `/content/dafx25_antialiasing_neural`
4. **Symlink cell:** Link Drive directories to repo:
   - `Drive/AA_Neural/weights/` → `repo/weights/`
   - `Drive/AA_Neural/audio_data/` → `repo/audio_data/`
   - `Drive/AA_Neural/checkpoints/` → `repo/lightning_logs/`
5. **Config cell:** Set training variables (config_idx, max_epochs, wandb_key)
6. **wandb cell:** Login to wandb with API key
7. **Train cell:** Execute `python train.py --config 3 --wandb`

### Dependency Installation

Colab doesn't support conda. Convert env.yaml to pip installs:

**Core packages:**
- PyTorch/torchvision/torchaudio (upgrade Colab's pre-installed version)
- pytorch-lightning
- wandb
- librosa, auraloss
- neural-amp-modeler

**Strategy:** Use pip install commands, leverage Colab's pre-installed PyTorch to minimize install time.

## Google Drive Organization

### Folder Structure

```
Google Drive/
└── AA_Neural/
    ├── weights/
    │   ├── NAM/
    │   │   └── Marshall JCM 800 2203/
    │   │       └── JCM800...700.nam
    │   └── Proteus_Tone_Packs/  (optional, add when needed)
    │       └── AmpPack1/
    │           └── MesaMiniRec_HighGain_DirectOut.json
    ├── audio_data/
    │   └── val_input.wav
    └── checkpoints/
        └── (auto-created by training)
```

### Initial Setup (One-time)

1. Create `AA_Neural/` folder in Google Drive root
2. Upload `weights/NAM/` directory from local repo
3. Upload `audio_data/val_input.wav`
4. `checkpoints/` creates automatically during first training run

### File Persistence Strategy

**Local to Colab VM (temporary, fast):**
- Cloned repository code
- Python packages (cached by Colab)

**Persisted to Drive (permanent, slower I/O):**
- Model weights (input)
- Audio validation files (input)
- Training checkpoints (output)

**Rationale:** Symlinks provide transparent access while keeping checkpoints persistent across Colab disconnects.

## Training Execution

### Running Training

Execute train.py as subprocess with real-time output:
```python
!python train.py --config 3 --wandb --max_epochs 100
```

PyTorch Lightning progress bars and metrics stream directly to notebook output.

### wandb Integration

**Setup (once per session):**
```python
import wandb
wandb.login(key='your-api-key')
```

**Automatic logging (handled by train.py):**
- Metrics logged every training step
- Audio samples uploaded periodically
- Model checkpoints tracked
- Validation plots saved

**Remote monitoring:**
- Close Colab tab, monitor from wandb.ai dashboard
- Mobile app support
- Email notifications (configure in wandb settings)

### Handling Disconnects

**Colab limitations:**
- Free tier: ~12 hour sessions, 90min idle timeout
- Sessions reset on disconnect

**Strategy:**
- Checkpoints auto-save to Drive (survives disconnects)
- With GPU: 100 epochs × 1-2 min = 100-200 mins (fits in one session)
- If needed later: add `--resume_from_checkpoint` support

### Error Handling

**On training crash:**
- wandb retains all logged metrics up to crash point
- Last checkpoint available in `Drive/AA_Neural/checkpoints/`
- Notebook output shows Python traceback
- Can resume from checkpoint (manual intervention)

## Post-Training Workflow

### Automatic Exports

train.py automatically on completion:
1. Loads best checkpoint (based on `val/nmr_mean_dB`)
2. Runs final validation
3. Exports model to JSON in `checkpoints/best_export/`

**Output location:** `Drive/AA_Neural/checkpoints/version_X/best_export/`

### Retrieving Results

**Via Colab interface:**
- Right-click folders in file browser → Download
- Or access directly from Drive on local machine

**Via wandb:**
- All metrics, plots, audio samples
- Compare runs across local and Colab training
- Download artifacts

### Session Management

**New training run:**
- Re-run notebook cells
- PyTorch Lightning auto-creates new `version_X` folder
- Previous runs preserved in Drive

**Continuing existing run:**
- Requires adding checkpoint resume logic (not implemented initially)
- Can add if needed for very long training runs

**Cleanup:**
- Colab VM auto-resets on disconnect
- Drive files persist until manually deleted
- Strategy: Keep successful checkpoints, remove failed runs

## Resource Considerations

**Colab Free Tier:**
- T4 GPU (16GB VRAM)
- ~12GB system RAM
- ~12 hour max session length

**Batch sizes (from config.py):**
- Train: 40 (GPU) vs 16 (CPU)
- Val: 16 (GPU) vs 8 (CPU)

**Expected performance:**
- Current: 32 min/epoch (local CPU)
- Colab T4: ~1-2 min/epoch (estimated 20-30x speedup)
- 100 epochs: ~2-3 hours total (fits comfortably in one session)

## Design Trade-offs

**Chosen: Minimal wrapper approach**

**Pros:**
- train.py unchanged, works identically local/Colab
- Simple to maintain
- Easy transition when local GPU acquired
- Git keeps code in sync

**Cons:**
- Less interactive than pure notebook cells
- Can't inspect variables mid-training
- Must restart training to change hyperparameters

**Alternative considered:** Inline notebook cells

**Why rejected:**
- Code divergence from train.py
- Hard to sync changes back to repository
- Loses ability to run same code locally

**Alternative considered:** Hybrid module import

**Why rejected:**
- Added complexity for minimal benefit
- Wrapper approach sufficient for current needs

## Future Enhancements

**Potential additions:**
1. Checkpoint resume support for multi-session training
2. Automatic email notification on training completion
3. Hyperparameter sweep integration
4. Multi-GPU support (Colab Pro)
5. Automatic best model download at end of training

**Not implementing initially:** Keep it simple, add complexity only when needed.
