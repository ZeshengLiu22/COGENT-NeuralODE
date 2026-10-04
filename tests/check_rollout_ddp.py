"""Run with Python to check real two-rank rollout reductions and empty shards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from test_rollout_artifacts import ForecastModel, SyntheticDataset
from training.evaluator import Evaluator


class DistributedForecast(ForecastModel):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, data):
        return super().forward(data) * self.scale


def _worker(rank, rendezvous, expected):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        for order, reference in expected:
            dataset = SyntheticDataset(order=order)
            model = DistributedDataParallel(DistributedForecast())
            evaluator = Evaluator(model, dataset.normalizer, torch.device("cpu"))
            metrics = evaluator.evaluate_full_rollout(dataset, known_steps=3)
            for key in reference:
                np.testing.assert_allclose(metrics[key], reference[key], rtol=1e-13, atol=1e-13, err_msg=key)
            seen = model.module.seen
            assert seen == list(range(rank, len(order), 2)), seen
            all_seen = [None, None]
            dist.all_gather_object(all_seen, seen)
            assert sorted(all_seen[0] + all_seen[1]) == list(range(len(order))), all_seen
            if len(order) == 1:
                assert all_seen[1] == [], all_seen
    finally:
        dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    expected = []
    for order in ((0, 1, 2), (0,)):
        dataset = SyntheticDataset(order=order)
        reference = Evaluator(DistributedForecast(), dataset.normalizer, torch.device("cpu")).evaluate_full_rollout(dataset, known_steps=3)
        expected.append((order, reference))
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(str(Path(tmp) / "rendezvous"), expected), nprocs=2, join=True)
    print("Two-rank rollout metrics match single-rank metrics for uneven and empty shards.")


if __name__ == "__main__":
    main()
