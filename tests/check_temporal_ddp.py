"""Two-process CPU check for graph-wise TC sampling and model RNG isolation.

Run explicitly with GLOO_SOCKET_IFNAME=lo python tests/check_temporal_ddp.py.
This check requires local sockets; ordinary unittest discovery does not run it.
"""

from datetime import timedelta
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.trainer import Trainer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(0.5))
        self.masks = []

    def forward(self, batch):
        mask = torch.nn.functional.dropout(torch.ones_like(batch.y_future), p=0.5)
        self.masks.append(mask.detach().clone())
        return self.slope * batch.t_future[0].view(1, -1, 1) * mask


class Normalizer:
    def inverse_state(self, value):
        return value


def check_rank(rank, root_name):
    root = Path(root_name)
    torch.set_num_threads(1)
    torch.manual_seed(77 + rank)
    dist.init_process_group('gloo', init_method=f'file://{root / "rendezvous"}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = torch.nn.parallel.DistributedDataParallel(Model())
        datasets = [SimpleNamespace(scenario_files=[root / f'{split}.npz'], future_len=4)
                    for split in ('train', 'val', 'test')]
        t = torch.arange(1., 5.).view(1, -1)
        batch = Batch.from_data_list([
            Data(y_future=(rank + graph + 1) * t.reshape(1, 4, 1).repeat(nodes, 1, 1),
                 t_future=t.clone(), num_nodes=nodes)
            for graph, nodes in enumerate((2, 3 + rank))
        ])
        trainers = []
        for enabled in (False, True):
            config = {
                'seed': 77,
                'dataset': {'data_dir': str(root)},
                'training': {'epochs': 1, 'lr': 0.0, 'train_horizon_mode': 'uniform_random',
                             'train_horizon_min': 4, 'train_horizon_curriculum': {'enabled': False},
                             'temporal_consistency': {'enabled': enabled, 'mode': 'random_pair_increment'}},
                'evaluation': {}, 'amp': {'mode': 'none'},
            }
            trainers.append(Trainer(model, [batch.clone()], [], [], *datasets, Normalizer(), config,
                                    torch.device('cpu'), root / f'rank{rank}', logging.getLogger('ddp')))
        initial_rng = torch.random.get_rng_state().clone()
        baseline = trainers[0].train_epoch(1)
        baseline_rng = torch.random.get_rng_state().clone()
        baseline_mask = model.module.masks[-1].clone()
        torch.random.set_rng_state(initial_rng)
        temporal = trainers[1].train_epoch(1)
        assert torch.equal(torch.random.get_rng_state(), baseline_rng)
        assert torch.equal(model.module.masks[-1], baseline_mask)
        assert len(model.module.masks) == 2, 'Exactly one forward per training batch'
        assert trainers[1].tc_generator.initial_seed() == 77 + rank
        assert temporal['train_norm_mse'] == baseline['train_norm_mse']
        assert temporal['train_tc_random_pair'] > 0
        assert temporal['train_loss'] == temporal['train_total_objective']
        assert abs(temporal['train_loss'] - temporal['train_state_objective'] - temporal['train_tc_weighted']) < 1e-5
        values = torch.tensor([temporal['train_loss'], temporal['train_tc_raw']])
        rank_values = [torch.empty_like(values) for _ in range(2)]
        dist.all_gather(rank_values, values)
        assert torch.equal(rank_values[0], rank_values[1])
        print(f'rank {rank}: graph-batched T2, private seed {77 + rank}, dropout RNG, objective all-reduce, one forward PASS', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='cogent-tc-ddp-') as root:
        for split in ('train', 'val', 'test'):
            (Path(root) / f'{split}.npz').touch()
        mp.spawn(check_rank, args=(root,), nprocs=2, join=True)
