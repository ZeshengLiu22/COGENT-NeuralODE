#!/usr/bin/env python3
"""Bounded real-data CUDA/DDP integration check; never a formal result.

Run with torchrun on four GPUs. Canonical science, model widths, data splits,
normalization and epoch series counts are preserved. Only the training iterator
is limited to four batches per rank and the run to one epoch. Validation/test
and standalone artifact evaluation still roll out every held-out scenario to end.
"""
from __future__ import annotations

import argparse
import gc
import os
from collections import Counter
from itertools import islice
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.factory import build_dataset, build_loader, build_splits
from datasets.normalization import FeatureNormalizer
from models import build_model
from training.horizon_sampling import synchronized_horizon
from training.trainer import Trainer
from utils import cleanup_distributed, configure_logging, load_config_bundle, save_json, seed_everything, setup_distributed


class SmokeLoader:
    """Limit consumed batches without changing the dataset or DDP sampler."""

    def __init__(self, loader, limit: int):
        self.loader = loader
        self.sampler = loader.sampler
        self.limit = min(len(loader), limit)

    def __len__(self):
        return self.limit

    def __iter__(self):
        return islice(self.loader, self.limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('issm', 'anuga'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-batches', type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires real CUDA devices; CPU fallback is prohibited.')
    if args.train_batches < 2:
        raise ValueError('At least two optimization steps are needed to exercise variable horizons.')
    runtime = setup_distributed('nccl')
    if runtime['world_size'] != 4:
        raise RuntimeError('Run this check with four GPUs using torchrun --nproc_per_node=4.')
    rank = int(runtime['rank'])
    device = torch.device('cuda', runtime['local_rank'])
    name = args.dataset
    future, known, budget, scenarios = (180, 60, 60, 28) if name == 'issm' else (64, 8, 9, 12)
    ablations = f'configs/ablations/{name}'
    stack = [
        'configs/default.yaml', f'configs/datasets/{name}.yaml',
        f'configs/protocols/{name}/main.yaml', 'configs/models/node2.yaml',
        f'{ablations}/history/h1.yaml', f'{ablations}/architecture/full.yaml',
        f'{ablations}/training_horizon/k{future}.yaml',
        f'{ablations}/rollout_start/known{known}.yaml',
        f'{ablations}/temporal_consistency/tc0.yaml', f'configs/runtime/{name}_fast.yaml',
    ]
    config = load_config_bundle([PROJECT_ROOT / path for path in stack])
    config['training']['epochs'] = 1
    # Diagnostic loader process settings; no science or model-width overrides.
    config['dataset']['num_workers'] = 0
    config['dataset']['persistent_workers'] = False
    seed_everything(int(config['seed']) + rank)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output / 'smoke.log' if rank == 0 else None)
    if rank == 0:
        save_json(output / 'config.json', config)
        (output / 'config_stack.txt').write_text('\n'.join(stack) + '\n')
        logger.info('CUDA smoke only: %s, four GPUs, at most %d batches/rank, one epoch', name, args.train_batches)
    train_files, val_files, test_files = build_splits(config)
    train = build_dataset(name, train_files, 'train', config)
    assert len(train.scenario_infos) == scenarios
    assert len(train) == scenarios * budget
    assert Counter(item.scenario_index for item in train.active_windows) == Counter({i: budget for i in range(scenarios)})
    normalizer = FeatureNormalizer.fit_from_trajectories(
        train.iter_trajectories(), std_floor=float(config['normalization']['std_floor']),
    )
    train.normalizer = normalizer
    val = build_dataset(name, val_files, 'val', config, normalizer)
    test = build_dataset(name, test_files, 'test', config, normalizer)
    assert not val.all_windows and not test.all_windows
    loader = build_loader(train, batch_size=int(config['training']['batch_size']), num_workers=0,
                          distributed=True, shuffle=True, pin_memory=True)
    assert loader.sampler.total_size == len(train)
    sample = train[0]
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1]).to(device)
    prediction_lengths = []

    def verify_series(module, inputs, prediction):
        if module.training:
            target = inputs[0].y_future
            assert prediction.shape == target.shape
            assert prediction.shape[1] <= future
            prediction_lengths.append(int(prediction.shape[1]))

    model.register_forward_hook(verify_series)
    model = DistributedDataParallel(model, device_ids=[runtime['local_rank']],
                                    find_unused_parameters=bool(config['distributed']['find_unused_parameters']))
    trainer = Trainer(model=model, train_loader=SmokeLoader(loader, args.train_batches),
                      train_dataset=train, val_dataset=val, test_dataset=test,
                      normalizer=normalizer, config=config, device=device, output_dir=output, logger=logger)
    horizons = []

    def record_horizon(*values, **kwargs):
        horizon = synchronized_horizon(*values, **kwargs)
        horizons.append(horizon)
        return horizon

    with patch('training.trainer.synchronized_horizon', record_horizon):
        summary = trainer.fit()
    assert len(horizons) == len(trainer.train_loader)
    assert horizons == prediction_lengths
    assert len(set(horizons)) > 1, f'No variable k_eff observed: {horizons}'
    rank_horizons = [None] * 4
    dist.all_gather_object(rank_horizons, horizons)
    assert all(values == horizons for values in rank_horizons)
    save_json(output / f'rank{rank}_training_evidence.json', {
        'device': str(device), 'gpu': torch.cuda.get_device_name(device),
        'epoch_training_series': len(train), 'series_per_scenario': budget,
        'ddp_padding': loader.sampler.total_size - len(train),
        'sampled_k_eff': horizons, 'complete_prediction_lengths': prediction_lengths,
        'formal_result': False,
    })
    dist.barrier()
    cleanup_distributed()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    if rank != 0:
        return
    checkpoint = output / 'best.pt'
    assert checkpoint.is_file()
    evaluation_output = output / 'standalone'
    distributed_keys = {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
    standalone_env = {key: value for key, value in os.environ.items()
                      if key not in distributed_keys and not key.startswith('TORCHELASTIC_')}
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / 'scripts/evaluate.py'),
        '--checkpoint', str(checkpoint), '--device', 'cuda:0', '--split', 'test',
        '--output-dir', str(evaluation_output), '--skip-anuga-export',
    ], cwd=PROJECT_ROOT, env=standalone_env, check=True)
    artifacts = list(evaluation_output.glob('*_predictions.npz'))
    assert len(artifacts) == 1, artifacts
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / 'scripts/postprocess_rollout.py'),
        '--artifacts', str(artifacts[0]), '--mode', 'full',
        '--output-prefix', str(output / 'postprocess'),
    ], cwd=PROJECT_ROOT, env=standalone_env, check=True)
    save_json(output / 'smoke_result.json', {
        'passed': True, 'formal_result': False, 'dataset': name,
        'devices': 4, 'canonical_model_widths': True, 'canonical_splits': True,
        'train_series_per_epoch': len(train), 'train_series_per_scenario_per_epoch': budget,
        'training_limit_batches_per_rank': args.train_batches,
        'horizons': horizons, 'rollout_known_steps': known,
        'best_checkpoint': str(checkpoint), 'artifact': str(artifacts[0]),
        'standalone_evaluation_and_postprocessing': 'passed', 'metrics': summary,
    })


if __name__ == '__main__':
    main()
