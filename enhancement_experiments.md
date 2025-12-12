# Antialiasing Neural Network Enhancement Experiments

Given the training time constraint (22 minutes per epoch, roughly 15 hours for 40 epochs), we need to be strategic. This document provides an evidence-based experimental roadmap.

---

## Phase 1: Baseline Measurement [COMPLETED]

### 1.1 Frequency Band Analysis Results

Phase 1 analysis tested 50 frequencies from 27.5 Hz to 12 kHz, computing NMR-S (aliasing) and HESR-R (harmonic preservation) metrics broken down by frequency band.

**Models Tested:**
- Teacher: `weights/NAM/.../JCM800 2203 - P5 B5 M5 T5 MV7 G10 - AZG - 700.nam`
- Student: `weights/antialiased/jhm8_tcn_student.nam`
- Sample rate: 48 kHz

**Raw Results:**

| Frequency Band | NMR-S Teacher | NMR-S Student | HESR-R Teacher | HESR-R Student |
|----------------|---------------|---------------|----------------|----------------|
| 20-500 Hz (Bass) | -39.02 dB | -44.88 dB | -115.42 dB | -5.50 dB |
| 500-2000 Hz (Midrange) | +3.11 dB | -33.80 dB | -114.80 dB | -7.97 dB |
| 2000-8000 Hz (Presence) | +31.86 dB | -19.11 dB | -116.76 dB | -9.34 dB |
| 8000-12000 Hz (High harmonics) | +40.23 dB | -3.86 dB | -119.15 dB | -19.02 dB |
| 12000-20000 Hz (Above training) | -117.66 dB | -118.36 dB | -119.03 dB | -11.35 dB |

**Metric Interpretation:**
- **NMR-S** (Noise-to-Mask Ratio): Lower is better. Values below -10 dB indicate inaudible aliasing.
- **HESR-R** (Harmonic Error Signal Ratio): Lower is better. Values near -20 dB or below indicate excellent harmonic preservation. Values above -10 dB indicate audible harmonic alteration.

### 1.2 Key Findings

#### Finding 1: Student Successfully Suppresses Aliasing

The student achieves substantial aliasing reduction across all bands where the teacher had significant aliasing:

| Band | Improvement | Student NMR-S | Verdict |
|------|-------------|---------------|---------|
| 500-2000 Hz | 37 dB | -33.80 dB | Well below -10 dB threshold |
| 2000-8000 Hz | 51 dB | -19.11 dB | Well below -10 dB threshold |
| 8000-12000 Hz | 44 dB | -3.86 dB | Near threshold, marginal |

**Conclusion:** Aliasing suppression objective is largely achieved, except marginally in 8-12 kHz.

#### Finding 2: Student Significantly Alters Harmonic Structure

The student's HESR-R values reveal substantial harmonic distortion:

| Band | Teacher HESR-R | Student HESR-R | Harmonic Error |
|------|----------------|----------------|----------------|
| 20-500 Hz | -115 dB | -5.50 dB | ~53% RMS difference |
| 500-2000 Hz | -115 dB | -7.97 dB | ~40% RMS difference |
| 2000-8000 Hz | -117 dB | -9.34 dB | ~34% RMS difference |
| 8000-12000 Hz | -119 dB | -19.02 dB | ~11% RMS difference |

**Conclusion:** The student is substantially altering harmonics, not just removing aliasing. This is audible.

#### Finding 3: Collateral Damage in 12-20 kHz Band

The 12-20 kHz band shows a critical anomaly:
- Teacher NMR-S: -117.66 dB (essentially zero aliasing)
- Student NMR-S: -118.36 dB (no improvement needed or achieved)
- Student HESR-R: -11.35 dB (significant harmonic alteration)

**Conclusion:** The student is damaging harmonics in a band where there was no aliasing to fix. This is pure collateral damage.

---

## Root Cause Analysis

### Cause 1: Pre-Emphasis LPF Cutoff at 12 kHz

The training loss function applies a lowpass filter before computing ESR and NMR:

```python
# From spectral.py line 210-213
filter_type == "lp":
    pb_edge = 12e3   # 12 kHz passband
    sb_edge = 16e3   # 16 kHz stopband
    As = 80          # 80 dB attenuation
```

**Effect:** Errors above 12 kHz are attenuated by up to 80 dB in the loss function. The model learns to ignore this region entirely, causing:
- No incentive to preserve harmonics above 12 kHz
- Collateral damage from whatever the model does in that region
- The -11.35 dB HESR-R in 12-20 kHz band

### Cause 2: λ=1.0 Over-Weights Aliasing Suppression

Current loss configuration:
```python
'loss_weights': {
    'esr_normal': 0.5,  # ESR weight
    'nmr': 0.5,         # NMR weight
    'dc': 1.0,
}
# Effective λ = nmr/esr = 1.0
```

**Effect:** With λ=1.0, the NMR term dominates wherever aliasing exists. The model aggressively suppresses aliasing but sacrifices harmonic accuracy in the process. Evidence:
- NMR-S improvements of 37-51 dB exceed what's needed for inaudibility (-10 dB threshold)
- HESR-R degradation of 100+ dB (from -115 dB to -5 to -10 dB) indicates severe harmonic alteration

### Cause 3: Insufficient f0 Training Range

Training data uses MIDI notes 21-127 (27.5 Hz to ~12.5 kHz):
```python
'train_data': {
    'midi_min': 21,  # 27.5 Hz
    # midi_max implicitly 127 = ~12.5 kHz
}
```

**Effect:** The model never sees inputs above 12.5 kHz during training. Combined with the pre-emphasis LPF, this creates a "blind spot" from 12-24 kHz.

---

## Revised Experimental Strategy

Based on Phase 1 evidence, the experiments are re-prioritized:

### Priority Matrix

| Experiment | Target Problem | Expected Impact | Training Time | Priority |
|------------|----------------|-----------------|---------------|----------|
| B: λ=0.3 | Harmonic preservation 0-12 kHz | High | 15 hours | **1 (Highest)** |
| A: Extended f0 + LPF | 12-20 kHz collateral damage | High | 15 hours | **2** |
| C: Curriculum λ | Both problems | Medium | 15 hours | **3** |
| D: Harmonic loss term | Harmonic preservation | Medium | 15 hours | **4** |
| Combined A+B | All problems | High | 15 hours | **5 (Final)** |

### Decision Tree

```
Start with Experiment B (λ=0.3)
    │
    ├── If HESR-R improves AND NMR-S stays below -10 dB:
    │   └── Success! Proceed to Experiment A to fix 12-20 kHz
    │
    ├── If HESR-R improves BUT NMR-S exceeds -10 dB:
    │   └── Try λ=0.5 (intermediate value)
    │
    └── If HESR-R doesn't improve:
        └── Problem is not λ. Try Experiment D (harmonic loss term)
```

---

## Phase 2: Single-Variable Experiments

### Experiment B: Lambda Sweep [NOW HIGHEST PRIORITY]

**Hypothesis:** λ=1.0 over-weights aliasing suppression, causing unnecessary harmonic distortion.

**Evidence from Phase 1:**
- NMR-S improvements far exceed the -10 dB threshold needed for inaudibility
- HESR-R degradation is severe (100+ dB worse than teacher)
- The 8-12 kHz band has the BEST harmonic preservation (-19 dB) despite having the WORST aliasing suppression (-3.86 dB), suggesting the trade-off is inverted

**Config Changes:**

For λ=0.3 (prioritize harmonics), edit `config.py`:
```python
'loss_weights': {
    'mesr': 0.0,
    'esr': 0.0,
    'asr': 0.0,
    'esr_normal': 0.769,  # 1.0 / (1.0 + 0.3)
    'nmr': 0.231,         # 0.3 / (1.0 + 0.3)
    'dc': 1.0,
},
```

For λ=0.1 (aggressive harmonic preservation), edit `config.py`:
```python
'loss_weights': {
    'mesr': 0.0,
    'esr': 0.0,
    'asr': 0.0,
    'esr_normal': 0.909,  # 1.0 / (1.0 + 0.1)
    'nmr': 0.091,         # 0.1 / (1.0 + 0.1)
    'dc': 1.0,
},
```

**Success Criteria:**
- NMR-S remains below -10 dB in all bands (aliasing stays inaudible)
- HESR-R improves from current -5 to -10 dB toward -15 to -20 dB (harmonics preserved)

**Expected Outcome:** λ=0.3 should provide sufficient aliasing suppression (NMR-S < -10 dB) while dramatically improving harmonic preservation.

---

### Experiment A: Extended Frequency Range [SECOND PRIORITY]

**Hypothesis:** The 12-20 kHz collateral damage is caused by:
1. Pre-emphasis LPF ignoring errors above 12 kHz
2. No training data with f0 above 12.5 kHz

**Evidence from Phase 1:**
- 12-20 kHz band shows -11.35 dB HESR-R despite no aliasing to fix
- This is pure collateral damage from the training process

**Config Changes:**

Step 1: Extend f0 range in `config.py`:
```python
'train_data': {
    'midi_min': 21,
    'linear_f0_sample': True,  # Use Hz directly instead of MIDI
    'f0_min': 27.5,            # 27.5 Hz
    'f0_max': 18000,           # 18 kHz (below Nyquist/2)
    'dB_min': -60,
    'dB_max': 0,
    'num_tones': 1600,
    'dur': 1.2,
},
```

Step 2: Extend pre-emphasis LPF in `spectral.py` (line 211-213):
```python
elif filter_type == "lp":
    As = 80
    pb_edge = 18e3   # Changed from 12e3
    sb_edge = 22e3   # Changed from 16e3
```

**Success Criteria:**
- HESR-R in 12-20 kHz band improves from -11.35 dB toward -15 to -20 dB
- NMR-S in other bands not significantly degraded

**Risk:** Extending the training range may spread model capacity thinner, potentially degrading performance in the critical 500-8000 Hz range.

---

### Experiment C: Curriculum Lambda Schedule [THIRD PRIORITY]

**Hypothesis:** A schedule that starts aggressive (high λ) and relaxes (low λ) could achieve both aliasing suppression and harmonic preservation.

**Rationale:**
- Early training: High λ teaches the model to suppress aliasing patterns
- Late training: Low λ refines harmonic accuracy

**Implementation:**

Add to `train.py` in the training loop:
```python
def get_lambda_weight(epoch, max_epochs=40):
    """
    Curriculum schedule: λ = 3.0 → 0.3 over training
    """
    lambda_start = 3.0
    lambda_end = 0.3
    # Exponential decay
    progress = epoch / max_epochs
    return lambda_start * (lambda_end / lambda_start) ** progress

# In loss_function or training step:
current_lambda = get_lambda_weight(self.current_epoch, conf['max_epochs'])
nmr_weight = current_lambda / (1.0 + current_lambda)
esr_weight = 1.0 / (1.0 + current_lambda)
```

**Success Criteria:**
- Combines benefits of high and low λ
- NMR-S < -10 dB AND HESR-R < -15 dB across all bands

**When to Try:** After Experiment B confirms λ affects the trade-off as expected.

---

### Experiment D: Harmonic Preservation Loss Term [FOURTH PRIORITY]

**Hypothesis:** Explicitly penalizing harmonic magnitude differences will preserve the distortion character.

**Implementation:**

Add to `train.py`:
```python
def harmonic_loss(y_stud, y_teach_bl, f0, fs, num_harmonics=10):
    """
    Penalize differences in harmonic magnitudes.
    """
    # Extract harmonic magnitudes using FFT
    N = y_stud.shape[-1]

    S_stud = torch.fft.rfft(y_stud)
    S_teach = torch.fft.rfft(y_teach_bl)

    # Get harmonic bins
    harmonics = f0 * torch.arange(1, num_harmonics + 1, device=y_stud.device)
    bins = torch.round(harmonics * N / fs).to(torch.int64)
    valid = bins < (N // 2)
    bins = bins[valid]

    # Extract magnitudes
    H_stud = S_stud[..., bins].abs()
    H_teach = S_teach[..., bins].abs()

    # Weight lower harmonics more heavily
    weights = 1.0 / torch.arange(1, len(bins) + 1, device=y_stud.device, dtype=torch.float)
    weights = weights / weights.sum()

    # MSE in dB domain
    H_stud_db = 20 * torch.log10(H_stud + 1e-8)
    H_teach_db = 20 * torch.log10(H_teach + 1e-8)

    return torch.sum(weights * (H_stud_db - H_teach_db) ** 2)
```

Update loss function:
```python
'loss_weights': {
    'esr_normal': 0.5,
    'nmr': 0.5,
    'harmonic': 0.3,  # New term
    'dc': 1.0,
},
```

**When to Try:** If Experiment B fails to improve HESR-R despite reducing λ.

---

## Phase 3: Recommended Experiment Order

### Run 1: Current Training [IN PROGRESS]
- Configuration: λ=1.0 (baseline)
- Purpose: Establishes baseline student performance
- Status: Phase 1 analysis completed, results documented above

### Run 2: Experiment B with λ=0.3 [NEXT]
- Configuration: `esr_normal: 0.769, nmr: 0.231`
- Purpose: Test if reduced λ improves harmonic preservation
- Time: 15 hours training + 1 hour evaluation
- Evaluation: Run Phase 1 analysis on new student model

### Run 3: Decision Point
- **If Run 2 succeeds (HESR-R < -15 dB, NMR-S < -10 dB):**
  → Proceed to Experiment A to fix 12-20 kHz
- **If Run 2 partially succeeds (HESR-R improves but NMR-S > -10 dB):**
  → Try λ=0.5 as intermediate
- **If Run 2 fails (HESR-R doesn't improve):**
  → Try Experiment D (harmonic loss term)

### Run 4: Experiment A (Extended f0 + LPF)
- Configuration: f0_max=18000, pb_edge=18000
- Purpose: Fix 12-20 kHz collateral damage
- Combine with best λ from Run 2/3

### Run 5: Final Combined Model
- Configuration: Best λ + Extended f0 range
- Purpose: Production-ready model
- Full evaluation including informal listening tests

---

## Phase 4: Evaluation Protocol

### 4.1 Automated Metrics (Per-Band)

Run `phase1_analysis.py` on each trained model:
```bash
cd .worktrees/phase1-analysis
conda activate aa_neural
python phase1_analysis.py --student path/to/new_student.nam
```

**Target Metrics:**

| Band | NMR-S Target | HESR-R Target |
|------|--------------|---------------|
| 20-500 Hz | < -10 dB | < -15 dB |
| 500-2000 Hz | < -10 dB | < -15 dB |
| 2000-8000 Hz | < -10 dB | < -15 dB |
| 8000-12000 Hz | < -10 dB | < -15 dB |
| 12000-20000 Hz | < -10 dB | < -15 dB |

### 4.2 Visual Inspection

Check `phase1_results/step1_2_high_freq_spectrograms.png`:
- Look for missing harmonic lines above 12 kHz (indicates suppression)
- Compare teacher vs student harmonic structure
- Verify aliasing "folding" patterns are reduced

### 4.3 Informal Listening

Before formal MUSHRA:
1. Guitar DI through teacher vs student
2. High notes (frets 12+) - check for "dullness"
3. Sine sweep - check for aliasing artifacts

**Red flags:**
- Student sounds "duller" or "less distorted" → harmonic preservation failing
- Student has "chirpy" or "metallic" artifacts → aliasing not fully suppressed

---

## Appendix: Quick Reference Config Changes

### Current Baseline (λ=1.0)
```python
'loss_weights': {
    'esr_normal': 0.5,
    'nmr': 0.5,
    'dc': 1.0,
},
```

### Experiment B: λ=0.3
```python
'loss_weights': {
    'esr_normal': 0.769,
    'nmr': 0.231,
    'dc': 1.0,
},
```

### Experiment B: λ=0.1
```python
'loss_weights': {
    'esr_normal': 0.909,
    'nmr': 0.091,
    'dc': 1.0,
},
```

### Experiment B: λ=3.0 (more aggressive aliasing)
```python
'loss_weights': {
    'esr_normal': 0.25,
    'nmr': 0.75,
    'dc': 1.0,
},
```

### Experiment A: Extended LPF (in spectral.py)
```python
# Line 211-213
pb_edge = 18e3   # Was 12e3
sb_edge = 22e3   # Was 16e3
```

---

## Summary

**Phase 1 revealed two distinct problems:**

1. **0-12 kHz: Over-aggressive aliasing suppression** (λ=1.0 is too high)
   - Solution: Experiment B (λ=0.3)

2. **12-20 kHz: Collateral damage from LPF** (loss function ignores this region)
   - Solution: Experiment A (extended LPF)

**Recommended path:**
1. Run Experiment B (λ=0.3) first - highest expected impact, single config change
2. Evaluate with Phase 1 analysis
3. If successful, add Experiment A to fix remaining 12-20 kHz issues
4. Combine best settings for final model

**Total time:** ~3-4 training runs (45-60 hours) + evaluation time.
