"""
Profile training to identify performance bottlenecks.

Usage:
    python profile_training.py --config 3 --num_batches 10

Then view trace.json in Chrome at chrome://tracing
"""

import argparse
import sys
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import pytorch_lightning as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OpenAmp'))

from config import get_config
from train import AARNN
from dataloader import CachedSineToneDataset


def profile_training(args):
    """Profile training loop to identify bottlenecks."""

    # Load config
    conf = get_config(args.config)
    print(f"Profiling config {args.config}: {conf['model_name']}")
    print(f"Using cached data: {args.use_cached_data}")
    print(f"Profiling {args.num_batches} batches...\n")

    # Create model
    model = AARNN(conf)
    model.double()

    # Create dataloader
    if args.use_cached_data:
        dataset = CachedSineToneDataset(args.cached_train_path)
    else:
        from dataloader import SineToneDataset
        dataset = SineToneDataset(conf)

    loader = DataLoader(
        dataset,
        batch_size=conf['batch_size']['train'],
        num_workers=0,  # Single-threaded for profiling
        shuffle=False
    )

    # Warmup batches
    warmup_batches = 2
    total_batches = warmup_batches + args.num_batches

    # Create trainer with profiler
    print("Warming up...")
    print("Starting profiling...\n")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        profiler='simple',  # Use Lightning's simple profiler for overview
    )

    # Custom profiling with PyTorch profiler
    class ProfilerCallback(pl.Callback):
        def __init__(self, warmup_batches, profile_batches):
            self.warmup_batches = warmup_batches
            self.profile_batches = profile_batches
            self.prof = None
            self.batch_count = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.batch_count == self.warmup_batches:
                # Start profiling after warmup
                print(f"Starting PyTorch profiler (batch {self.batch_count})...")
                self.prof = profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True
                )
                self.prof.__enter__()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.batch_count += 1
            if self.batch_count == self.warmup_batches + self.profile_batches:
                # Stop profiling
                print(f"Stopping PyTorch profiler (batch {self.batch_count})...")
                if self.prof:
                    self.prof.__exit__(None, None, None)
                    self._print_results()
                else:
                    print("Warning: profiler was not started")
                trainer.should_stop = True

        def _print_results(self):
            prof = self.prof

            print("\n" + "=" * 80)
            print("TOP 30 OPERATIONS BY CPU TIME")
            print("=" * 80)
            print(prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=30
            ))

            print("\n" + "=" * 80)
            print("TOP 20 OPERATIONS BY MEMORY")
            print("=" * 80)
            print(prof.key_averages().table(
                sort_by="self_cpu_memory_usage",
                row_limit=20
            ))

            print("\n" + "=" * 80)
            print("AGGREGATED BY FUNCTION")
            print("=" * 80)

            # Group by major functions
            key_averages = prof.key_averages(group_by_stack_n=5)

            # Filter for our code
            our_functions = [
                item for item in key_averages
                if any(name in item.key for name in [
                    'training_step', 'bandlimit_batch', 'nmr',
                    'cheb_fft', 'synthesise_batch', 'forward'
                ])
            ]

            if our_functions:
                print("\nOur code breakdown:")
                for item in sorted(our_functions, key=lambda x: x.cpu_time_total, reverse=True)[:15]:
                    print(f"{item.key:50s} {item.cpu_time_total/1000:.2f}ms")

            # Export trace
            trace_path = "trace.json"
            prof.export_chrome_trace(trace_path)
            print(f"\n✓ Trace exported to {trace_path}")
            print(f"  View in Chrome at: chrome://tracing")

            # Summary statistics
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)

            total_time_ms = sum(item.cpu_time_total for item in key_averages) / 1000
            print(f"Total profiled time: {total_time_ms:.2f}ms ({total_time_ms/self.profile_batches:.2f}ms per batch)")

            # Find top bottlenecks
            print("\nTop 5 bottlenecks:")
            all_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)
            for idx, item in enumerate(all_ops[:5], 1):
                pct = 100 * item.cpu_time_total / (total_time_ms * 1000) if total_time_ms > 0 else 0
                print(f"  {idx}. {item.key[:60]:60s} {item.cpu_time_total/1000:8.2f}ms ({pct:5.1f}%)")

            print("\n" + "=" * 80)
            print("ACTIONABLE INSIGHTS")
            print("=" * 80)

            # Analyze results
            insights = []

            # Check FFT operations
            fft_time = sum(item.cpu_time_total for item in key_averages if 'fft' in item.key.lower())
            if total_time_ms > 0 and fft_time / (total_time_ms * 1000) > 0.1:
                insights.append(f"FFT operations: {fft_time/1000:.1f}ms ({100*fft_time/(total_time_ms*1000):.1f}%) - Consider reducing FFT size or frequency")

            # Check matmul/linear operations
            matmul_time = sum(item.cpu_time_total for item in key_averages if any(x in item.key.lower() for x in ['matmul', 'linear', 'mm', 'addmm']))
            if total_time_ms > 0 and matmul_time / (total_time_ms * 1000) > 0.15:
                insights.append(f"Matrix operations: {matmul_time/1000:.1f}ms ({100*matmul_time/(total_time_ms*1000):.1f}%) - Model forward/backward pass")

            # Check convolution operations
            conv_time = sum(item.cpu_time_total for item in key_averages if 'conv' in item.key.lower())
            if total_time_ms > 0 and conv_time / (total_time_ms * 1000) > 0.1:
                insights.append(f"Convolution operations: {conv_time/1000:.1f}ms ({100*conv_time/(total_time_ms*1000):.1f}%)")

            # Check memory operations
            mem_time = sum(item.cpu_time_total for item in key_averages if any(x in item.key.lower() for x in ['clone', 'copy', 'contiguous']))
            if total_time_ms > 0 and mem_time / (total_time_ms * 1000) > 0.05:
                insights.append(f"Memory operations: {mem_time/1000:.1f}ms ({100*mem_time/(total_time_ms*1000):.1f}%) - Reduce tensor copies")

            if insights:
                for insight in insights:
                    print(f"  • {insight}")
            else:
                print("  • Profile more batches for clearer patterns")

            print("\n" + "=" * 80)

    # Add profiler callback and run training
    profiler_callback = ProfilerCallback(warmup_batches, args.num_batches)
    trainer.callbacks.append(profiler_callback)
    trainer.fit(model, loader)


def main():
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--config", type=int, default=3, help="Model config index")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--use_cached_data", action="store_true", help="Use pre-cached teacher outputs")
    parser.add_argument("--cached_train_path", type=str, default="cached_train.pt", help="Path to cached training data")

    args = parser.parse_args()

    print("=" * 80)
    print("PyTorch Training Profiler")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU (Intel Mac Pro)")
    print("=" * 80 + "\n")

    profile_training(args)

    print("\nNext steps:")
    print("  1. View trace.json in chrome://tracing for detailed timeline")
    print("  2. Identify top bottleneck operations")
    print("  3. Focus optimization efforts on highest-impact areas")
    print("  4. Re-profile after optimizations to measure improvement")


if __name__ == '__main__':
    main()
