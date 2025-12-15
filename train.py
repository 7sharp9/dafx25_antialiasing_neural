import copy
import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenAmp'))
from Open_Amp.amp_model import AmpModel

from dataloader import SineToneDataset, SequenceDataset
from spectral import cheb_fft, bandlimit_batch, PerceptualFIRFilter
from nmr import NMR
from config import get_config

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--config", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)  # Override config
    parser.add_argument("--fast_dev_run", action="store_true")  # Quick debug run
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)  # Resume from checkpoint
    return parser.parse_args()


def dB10(x):
    """Convert linear ratio to decibels."""
    return 10 * torch.log10(x.clamp(min=1e-10))


def linear10(x):
    """Convert decibels to linear ratio."""
    return 10 ** (x / 10)


class AARNN(LightningModule):
    """Anti-Aliasing RNN trainer using teacher-student fine-tuning."""
    
    def __init__(self, conf: dict):
        super().__init__()
        self.save_hyperparameters(conf)
        self.conf = conf
        
        # Model setup
        self.model = AmpModel(conf['model_json'], conf['model_name'])
        self.model.double()
        
        # Loss components
        self.nmr = NMR(fs=conf['sample_rate'])
        self.aweight_fir = PerceptualFIRFilter(filter_type='aw')
        
        # Pre-emphasis filter
        if conf['pre_emph'] == 'lp':
            self.lpf = PerceptualFIRFilter(filter_type='lp')
        elif conf['pre_emph'] == 'aw':
            self.lpf = PerceptualFIRFilter(filter_type='aw')
        else:
            self.lpf = torch.nn.Identity()
        
        # EMA model for stable validation
        self.ema_model: Optional[AmpModel] = None
        self.ema_decay = conf.get('ema_decay', 0.999)
        
        # Manual optimization for TBPTT
        self.automatic_optimization = False
        
        # Validation state
        self._reset_val_metrics()
        
        # Timing
        self._batch_load_start = time.time()
        self._audio_log_counter = 0

    def _reset_val_metrics(self):
        """Initialize fresh validation metric accumulators."""
        self.val_metrics = {
            'mesr': {},
            'esr': {},
            'asr': {},
            'esr_normal': {},
            'esr_lpf': {},
            'nmr': {},
            'audio_esr': {},
        }

    def _update_ema(self):
        """Update exponential moving average of model weights."""
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
        else:
            with torch.no_grad():
                for ema_p, p in zip(self.ema_model.parameters(), 
                                    self.model.parameters()):
                    ema_p.lerp_(p, 1 - self.ema_decay)

    def _compute_loss(self, targ, pred, f0):
        """
        Compute training loss combining ESR and NMR.
        
        Args:
            targ: Bandlimited teacher output
            pred: Student model output
            f0: Fundamental frequencies for the batch
            
        Returns:
            loss: Scalar loss tensor
            metrics: Dict of metric values for logging
        """
        # Apply pre-emphasis filter
        targ_filt = self.lpf(targ)
        pred_filt = self.lpf(pred)
        
        loss = pred.new_zeros(1)
        metrics = {}
        
        weights = self.conf['loss_weights']
        
        # NMR loss (perceptually-weighted spectral)
        if weights['nmr'] > 0:
            nmr, nmr_dB = self.nmr(pred_filt, targ_filt)
            nmr_mean = linear10(nmr_dB.mean())
            loss = loss + weights['nmr'] * nmr_mean
            metrics['nmr'] = nmr_mean.detach()
            metrics['nmr_dB'] = nmr_dB.mean().detach()
        
        # ESR loss (time-domain)
        if weights['esr_normal'] > 0:
            esr = torch.sum((pred_filt - targ_filt) ** 2) / torch.sum(targ_filt ** 2)
            loss = loss + weights['esr_normal'] * esr
            metrics['esr'] = esr.detach()
        
        # DC loss
        if weights['dc'] > 0:
            dc = torch.mean(targ_filt - pred_filt) ** 2 / torch.mean(targ_filt ** 2)
            loss = loss + weights['dc'] * dc
            metrics['dc'] = dc.detach()
        
        # Optional bandlimited losses (disabled by default in config)
        if weights.get('mesr', 0) > 0 or weights.get('asr', 0) > 0:
            y_pred_bl, aliases = bandlimit_batch(pred.squeeze(-1), f0, self.conf['sample_rate'])
            
            if weights.get('mesr', 0) > 0:
                Y_bl = cheb_fft(targ).abs()
                Y_pred_bl = cheb_fft(y_pred_bl, dim=1).abs()
                mesr = torch.sum((Y_pred_bl - Y_bl) ** 2) / torch.sum(Y_bl ** 2)
                loss = loss + weights['mesr'] * mesr
                metrics['mesr'] = mesr.detach()
            
            if weights.get('asr', 0) > 0:
                asr = torch.sum(aliases ** 2) / torch.sum(y_pred_bl ** 2)
                loss = loss + weights['asr'] * asr
                metrics['asr'] = asr.detach()
        
        metrics['total'] = loss.detach()
        return loss, metrics

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        x, y, f0, dB = batch
        
        # Log batch loading time
        batch_load_time = time.time() - self._batch_load_start
        
        # Prepare target: bandlimit teacher output to remove aliasing
        y_bl, _ = bandlimit_batch(y.squeeze(-1), f0, self.conf['sample_rate'])
        warmup_samples = x.shape[1] - y_bl.shape[-1]
        
        # Handle model-specific warmup
        if self.model.model_class == 'SimpleRNN':
            self.model.model.reset_state()
            with torch.no_grad():
                self.model(x[:, :warmup_samples, :])
            x = x[:, warmup_samples:, :]
            warmup_samples = 0
        
        # TBPTT: process in frames
        tbptt_steps = self.conf['tbptt_steps']
        num_frames = y_bl.shape[-1] // tbptt_steps
        
        frame_metrics = []
        
        for frame_idx in range(num_frames):
            opt.zero_grad()
            
            start = tbptt_steps * frame_idx
            end = tbptt_steps * (frame_idx + 1)
            
            # Input includes warmup for WaveNet-style models
            x_frame = x[:, start:warmup_samples + end, :]
            y_frame = y_bl[:, start:end]
            
            # Forward pass
            y_pred = self.model(x_frame).squeeze(-1)
            y_pred = y_pred[:, -tbptt_steps:]  # Take only current frame
            
            # Compute loss
            loss, metrics = self._compute_loss(y_frame, y_pred, f0)
            
            # Backward pass
            self.manual_backward(loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            opt.step()
            
            # Detach RNN state for TBPTT
            if self.model.model_class == 'SimpleRNN':
                self.model.model.detach_state()
            
            frame_metrics.append(metrics)
        
        # Update EMA after each batch
        self._update_ema()
        
        # Aggregate frame metrics
        avg_metrics = {}
        for key in frame_metrics[0].keys():
            avg_metrics[f'train/{key}'] = torch.stack(
                [m[key] for m in frame_metrics]
            ).mean()
        
        avg_metrics['train/batch_load_time'] = batch_load_time
        avg_metrics['train/lr'] = opt.param_groups[0]['lr']
        
        self.log_dict(avg_metrics, on_step=True, on_epoch=False, prog_bar=True)
        
        self._batch_load_start = time.time()
        
        return loss

    def on_train_epoch_end(self):
        """Step LR scheduler at epoch end."""
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def _forward_with_state_handling(self, model, x, frame_size, detach_states=False):
        """
        Forward pass with proper state handling for RNNs.
        
        Args:
            model: The model to use (self.model or self.ema_model)
            x: Input tensor
            frame_size: Frame size for processing
            detach_states: Whether to detach states between frames
            
        Returns:
            Model output
        """
        if model.model_class == 'WaveNet':
            return model(x)
        
        # RNN: process in frames
        model.model.reset_state()
        num_frames = x.shape[1] // frame_size
        outputs = []
        
        for n in range(num_frames):
            start = frame_size * n
            end = frame_size * (n + 1)
            x_frame = x[:, start:end, :]
            y_frame = model(x_frame)
            outputs.append(y_frame)
            
            if detach_states:
                model.model.detach_state()
        
        return torch.cat(outputs, dim=1)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Use EMA model if available
        model = self.ema_model if self.ema_model is not None else self.model
        
        if dataloader_idx == 0:
            # Sine tone validation
            self._validate_sine_tones(model, batch, batch_idx)
        else:
            # Audio validation
            self._validate_audio(model, batch, batch_idx)

    def _validate_sine_tones(self, model, batch, batch_idx):
        """Validate on sine tone dataset."""
        x, y, f0, dB = batch
        
        # Bandlimit target
        y_bl, _ = bandlimit_batch(y.squeeze(-1), f0, self.conf['sample_rate'])
        
        # Forward pass with state detachment (matching training)
        y_pred = self._forward_with_state_handling(
            model, x, self.conf['tbptt_steps'], detach_states=True
        ).squeeze(-1)
        
        # Trim to match target length
        y_pred = y_pred[:, -self.conf['sample_rate']:]
        
        # Bandlimit prediction for comparison
        y_pred_bl, aliases = bandlimit_batch(y_pred, f0, self.conf['sample_rate'])
        
        # Compute spectra
        Y_bl = cheb_fft(y_bl).abs()
        Y_pred = cheb_fft(y_pred).abs()
        Y_pred_bl = cheb_fft(y_pred_bl, dim=1).abs()
        
        # Compute metrics per sample
        batch_size = x.shape[0]
        
        nmr, _ = self.nmr(y_pred, y_bl)
        mesr = torch.sum((Y_pred_bl - Y_bl) ** 2, dim=-1) / torch.sum(Y_bl ** 2, dim=-1)
        esr = torch.sum((y_pred_bl - y_bl) ** 2, dim=-1) / torch.sum(y_bl ** 2, dim=-1)
        asr = torch.sum(aliases ** 2, dim=-1) / torch.sum(y_pred_bl ** 2, dim=-1)
        esr_normal = torch.sum((y_pred - y_bl) ** 2, dim=-1) / torch.sum(y_bl ** 2, dim=-1)
        esr_lpf = torch.sum(self.lpf(y_pred - y_bl) ** 2, dim=-1) / torch.sum(self.lpf(y_bl) ** 2, dim=-1)
        
        # Store by f0 index
        for b in range(batch_size):
            f0_idx = int(f0[b].squeeze().item())
            self.val_metrics['esr'][f0_idx] = esr[b]
            self.val_metrics['mesr'][f0_idx] = mesr[b]
            self.val_metrics['asr'][f0_idx] = asr[b]
            self.val_metrics['esr_normal'][f0_idx] = esr_normal[b]
            self.val_metrics['esr_lpf'][f0_idx] = esr_lpf[b]
            self.val_metrics['nmr'][f0_idx] = nmr[b]

    def _validate_audio(self, model, batch, batch_idx):
        """Validate on audio dataset."""
        x, y = batch
        
        # Forward pass
        y_pred = self._forward_with_state_handling(
            model, x, self.conf['tbptt_steps'], detach_states=True
        )
        
        # Remove warmup region
        y = y[:, self.conf['tbptt_steps']:, :]
        y_pred = y_pred[:, self.conf['tbptt_steps']:, :]
        
        # Log audio samples periodically
        if self._audio_log_counter % 4 == 0 and batch_idx < 5:
            if self.logger is not None:
                self.logger.experiment.log({
                    f'audio/clip_{batch_idx}': wandb.Audio(
                        y_pred[-1, :, 0].cpu().numpy(),
                        sample_rate=self.conf['sample_rate']
                    ),
                    'epoch': self.current_epoch
                })
        
        if batch_idx == 0:
            self._audio_log_counter += 1
        
        # Compute A-weighted ESR
        y_weighted = self.aweight_fir(y)
        y_pred_weighted = self.aweight_fir(y_pred)
        esr = torch.sum((y_pred_weighted - y_weighted) ** 2) / torch.sum(y_weighted ** 2)
        
        self.val_metrics['audio_esr'][batch_idx] = esr

    def on_validation_epoch_end(self):
        """Aggregate and log validation metrics."""
        metrics = {'epoch': self.current_epoch}
        
        # Get f0 values for plotting
        f0_values = np.array(list(self.val_metrics['esr'].keys()))
        
        # Aggregate each metric
        for metric_name, metric_dict in self.val_metrics.items():
            if not metric_dict:
                continue
                
            values = torch.stack(list(metric_dict.values()))
            
            # Linear domain stats
            metrics[f'val/{metric_name}_mean'] = values.mean()
            metrics[f'val/{metric_name}_max'] = values.max()
            
            # dB domain stats
            metrics[f'val/{metric_name}_mean_dB'] = dB10(values.mean())
            metrics[f'val/{metric_name}_max_dB'] = dB10(values.max())
        
        # Frequency band breakdown for NMR
        if self.val_metrics['nmr']:
            nmr_values = torch.stack(list(self.val_metrics['nmr'].values()))
            bands = [
                ('bass', 27.5, 200),
                ('mid', 200, 1000),
                ('high', 1000, 4000),
                ('ultra', 4000, 22050),
            ]
            for name, lo, hi in bands:
                mask = (f0_values >= lo) & (f0_values < hi)
                if mask.any():
                    metrics[f'val/nmr_{name}_dB'] = dB10(nmr_values[mask].mean())
        
        # Create summary plot
        if len(f0_values) > 0 and self.val_metrics['nmr']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric_name in ['nmr', 'asr', 'esr_normal']:
                if self.val_metrics[metric_name]:
                    values = torch.stack(list(self.val_metrics[metric_name].values()))
                    ax.semilogx(f0_values, dB10(values).cpu().numpy(), label=metric_name)
            
            ax.axhline(y=-10, color='k', linestyle='--', alpha=0.5, label='Audibility threshold')
            ax.set_ylim([-120, 0])
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('[dB]')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if self.logger is not None:
                self.logger.experiment.log({'val/metrics_vs_f0': wandb.Image(fig)})
            
            plt.close(fig)
        
        # Log all metrics
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        # Export model checkpoint - only if using wandb logger
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            # Use the dir property correctly
            log_dir = self.logger.save_dir
            json_path = os.path.join(log_dir, 'json')
            os.makedirs(json_path, exist_ok=True)
            self.model.export(dir=json_path, to_append=f'_epoch={self.current_epoch}')
        
        # Reset for next epoch
        self._reset_val_metrics()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.conf['lr'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.conf['lr'] * 0.01,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def create_dataloaders(conf, args):
    """Create train, validation, and audio dataloaders."""
    
    if torch.cuda.is_available():
        num_workers = min(10, os.cpu_count() or 1)
        persistent_workers = True
        pin_memory = True
    else:
        num_workers = 0
        persistent_workers = False
        pin_memory = False
    
    # Training dataloader
    train_dataset = SineToneDataset(
        device=conf['model_json'],
        sample_rate=conf['sample_rate'],
        **conf['train_data']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf['batch_size']['train'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    # Validation dataloader (sine tones)
    val_dataset = SineToneDataset(
        device=conf['model_json'],
        sample_rate=conf['sample_rate'],
        **conf['val_data']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=conf['batch_size']['val'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=False,
    )
    
    # Audio validation dataloader
    in_audio, audio_sr = torchaudio.load('audio_data/val_input.wav')
    
    if audio_sr != conf['sample_rate']:
        from scipy.signal import resample
        new_length = int(conf['sample_rate'] / audio_sr * in_audio.shape[-1])
        in_audio = torch.from_numpy(
            resample(in_audio.numpy().squeeze(), new_length)
        ).unsqueeze(0)
    
    gain = 10 ** (conf['audio_val_data']['dB'] / 20)
    in_audio = gain * in_audio.to(torch.double)
    
    audio_dataset = SequenceDataset(
        input=in_audio,
        device=conf['model_json'],
        sequence_length=int(conf['audio_val_data']['dur'] * conf['sample_rate'])
    )
    audio_loader = DataLoader(
        audio_dataset,
        batch_size=conf['batch_size']['audio_val'],
    )
    
    return train_loader, val_loader, audio_loader


def create_callbacks(conf, use_wandb: bool):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val/nmr_mean_dB',
        mode='min',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{val_nmr_mean_dB:.2f}',
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping (conservative patience)
    early_stop_callback = EarlyStopping(
        monitor='val/nmr_mean_dB',
        mode='min',
        patience=20,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    if use_wandb:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    return callbacks


def main():
    args = parse_args()
    
    # Load configuration
    conf = get_config(args.config)
    
    # Override max_epochs if specified
    if args.max_epochs is not None:
        conf['max_epochs'] = args.max_epochs
    
    print(f"Configuration: {conf['model_name']}")
    print(f"Model: {conf['model_json']}")
    print(f"CPU cores: {os.cpu_count()}")
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create model
    model = AARNN(conf)
    print(model.model)
    
    # Create dataloaders
    train_loader, val_loader, audio_loader = create_dataloaders(conf, args)
    
    # Setup logging
    if args.wandb:
        # Determine save directory (Drive on Colab, local otherwise)
        import os
        if os.path.exists('/content/drive/MyDrive'):
            # Running on Colab with Drive mounted
            save_dir = '/content/drive/MyDrive/AA_Neural/checkpoints'
        else:
            # Running locally
            save_dir = None  # Use current directory

        logger = pl.loggers.WandbLogger(
            project='aa_rnn',
            save_dir=save_dir,
            config=conf,
            log_model=True,
        )
    else:
        logger = None
    
    # Create callbacks
    callbacks = create_callbacks(conf, args.wandb)
    
    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=conf['max_epochs'],
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )
    
    # Initial validation
    print("Running initial validation...")
    trainer.validate(model=model, dataloaders=[val_loader, audio_loader])
    
    # Training
    print("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=[val_loader, audio_loader],
        ckpt_path=args.resume_from_checkpoint,
    )
    
    # Final validation on best model instead of test
    if trainer.checkpoint_callback.best_model_path:
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        best_model = AARNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            conf=conf,
        )
        
        # Run final validation with best model
        print("Running final validation on best model...")
        trainer.validate(model=best_model, dataloaders=[val_loader, audio_loader])
        
        # Export best model
        export_dir = os.path.join(
            os.path.dirname(trainer.checkpoint_callback.best_model_path),
            'best_export'
        )
        os.makedirs(export_dir, exist_ok=True)
        best_model.model.export(dir=export_dir, to_append='_best')
        print(f"Exported best model to {export_dir}")


if __name__ == '__main__':
    main()