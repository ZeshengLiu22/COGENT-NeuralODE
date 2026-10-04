"""Dataset-specific formal controls, merge order, and configuration ownership."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.io import deep_update, load_config_bundle, load_yaml

HORIZONS = {'issm': (30, 45, 60, 75, 90, 120, 150, 180),
            'anuga': (8, 16, 24, 32, 40, 48, 56, 64)}


def canonical(dataset='issm', *overlays):
    return load_config_bundle([ROOT / 'configs' / path for path in (
        'default.yaml', f'datasets/{dataset}.yaml', f'protocols/{dataset}/main.yaml',
        'models/node2.yaml', *overlays, f'runtime/{dataset}_fast.yaml')])


class ConfigTest(unittest.TestCase):
    def test_recursive_merge_preserves_siblings_and_inputs(self):
        base = {'section': {'list': [1, 2], 'keep': 3, 'nested': {'x': 1, 'y': 2}}}
        original = deepcopy(base)
        merged = deep_update(base, {'section': {'list': [9], 'nested': {'x': 4}}})
        self.assertEqual(merged, {'section': {'list': [9], 'keep': 3, 'nested': {'x': 4, 'y': 2}}})
        self.assertEqual(base, original)

    def test_left_to_right_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            layers = ('default', 'dataset', 'protocol', 'model', 'history', 'architecture',
                      'training_horizon', 'rollout_start', 'temporal_consistency', 'runtime')
            for index, name in enumerate(layers):
                path = Path(tmp) / f'{name}.yaml'
                path.write_text(f'section:\n  value: {index}\n  {name}: true\n  list: [{index}]\n')
                paths.append(path)
            merged = load_config_bundle(paths)['section']
            self.assertEqual(merged['value'], len(layers) - 1)
            self.assertEqual(merged['list'], [len(layers) - 1])
            self.assertTrue(all(merged[name] for name in layers))

    def test_canonical_protocols_and_epoch_budgets(self):
        for dataset, k, start, scale, budget, batch, minimum, loss_scale, scenarios in (
            ('issm', 180, 60, 180., 60, 8, 24, 100., 28),
            ('anuga', 64, 8, 65., 9, 1, 8, 1., 12)):
            with self.subTest(dataset=dataset):
                cfg = canonical(dataset)
                self.assertEqual((cfg['dataset']['history_len'], cfg['dataset']['future_len'],
                                  cfg['evaluation']['known_steps'], cfg['model']['relative_time_scale']),
                                 (1, k, start, scale))
                self.assertEqual(cfg['dataset']['train_series_per_scenario_per_epoch'], budget)
                self.assertEqual((scenarios * budget) % 4, 0)
                self.assertEqual(cfg['model']['latent_dim'], 96)
                self.assertEqual(cfg['model']['history_encoder']['history_encoder_type'], 'transformer')
                self.assertTrue(all(cfg['model'][key] for key in (
                    'use_residual_decoder', 'use_history_in_ode', 'use_relative_time')))
                self.assertEqual(cfg['training']['batch_size'], batch)
                self.assertEqual(cfg['training']['train_horizon_min'], minimum)
                self.assertEqual(cfg['training']['loss_scale_factor'], loss_scale)
                self.assertIsNone(cfg['training']['train_horizon_max'])
                self.assertFalse(cfg['training']['temporal_consistency']['enabled'])
        self.assertEqual(canonical()['dataset']['split'], {
            'strategy': 'issm_rate_modulo', 'modulo': 20, 'val_remainder': 0, 'test_remainder': 10})
        self.assertEqual(canonical('anuga')['dataset']['split'], {
            'strategy': 'random', 'train': .6, 'val': .2, 'test': .2, 'seed': 42})
        self.assertEqual(canonical('anuga')['dataset']['file_patterns'], ['sim_*_merged.npz'])

    def test_explicit_controls_equal_canonical(self):
        for dataset, k, start in (('issm', 180, 60), ('anuga', 64, 8)):
            controls = ('history/h1.yaml', 'architecture/full.yaml', f'training_horizon/k{k}.yaml',
                        f'rollout_start/known{start}.yaml', 'temporal_consistency/tc0.yaml')
            self.assertEqual(canonical(dataset), canonical(
                dataset, *(f'ablations/{dataset}/{path}' for path in controls)))

    def test_temporal_overlays_only_change_their_axis(self):
        for dataset in ('issm', 'anuga'):
            for directory, values, prefix, section, key in (
                ('history', range(1, 9), 'h', 'dataset', 'history_len'),
                ('training_horizon', HORIZONS[dataset], 'k', 'dataset', 'future_len'),
                ('rollout_start', (60, 90, 120) if dataset == 'issm' else (8,),
                 'known', 'evaluation', 'known_steps')):
                paths = list((ROOT / f'configs/ablations/{dataset}/{directory}').glob('*.yaml'))
                self.assertEqual({p.stem for p in paths}, {f'{prefix}{value}' for value in values})
                for value in values:
                    path = f'ablations/{dataset}/{directory}/{prefix}{value}.yaml'
                    with self.subTest(path=path):
                        self.assertEqual(load_yaml(ROOT / 'configs' / path), {section: {key: value}})
                        expected = canonical(dataset)
                        expected[section][key] = value
                        self.assertEqual(canonical(dataset, path), expected)
                        self.assertGreaterEqual(expected['dataset']['future_len'],
                                                expected['training']['train_horizon_min'])

    def test_architecture_is_explicit_complete_factorial(self):
        expected = set(product(('transformer', 'lstm'), (True, False), (True, False), (True, False)))
        for dataset in ('issm', 'anuga'):
            base = ROOT / f'configs/ablations/{dataset}/architecture'
            variants = sorted(base.glob('a*.yaml'))
            self.assertEqual(len(variants), 16)
            actual = set()
            for index, path in enumerate(variants, 1):
                self.assertTrue(path.stem.startswith(f'a{index:02}_'))
                cfg = load_yaml(path)
                self.assertEqual(set(cfg), {'model'})
                model = cfg['model']
                self.assertEqual(set(model), {'history_encoder', 'use_residual_decoder',
                                              'use_history_in_ode', 'use_relative_time'})
                self.assertEqual(set(model['history_encoder']), {'history_encoder_type'})
                actual.add((model['history_encoder']['history_encoder_type'], model['use_residual_decoder'],
                            model['use_history_in_ode'], model['use_relative_time']))
            self.assertEqual(actual, expected)
            self.assertEqual(load_yaml(base / 'full.yaml'), load_yaml(variants[0]))

    def test_runtime_is_safe_and_contains_only_loader_settings(self):
        cfg = load_yaml(ROOT / 'configs/runtime/fast.yaml')
        self.assertEqual(set(cfg), {'dataset'})
        self.assertEqual(set(cfg['dataset']), {'num_workers', 'pin_memory', 'prefetch_factor', 'persistent_workers'})
        self.assertFalse(load_yaml(ROOT / 'configs/default.yaml')['dataset']['cache_in_memory'])
        for dataset in ('issm', 'anuga'):
            generic = load_config_bundle([ROOT / 'configs' / path for path in (
                'default.yaml', f'datasets/{dataset}.yaml', f'protocols/{dataset}/main.yaml',
                'runtime/fast.yaml')])
            self.assertFalse(generic['dataset']['cache_in_memory'])
            formal = load_yaml(ROOT / f'configs/runtime/{dataset}_fast.yaml')
            self.assertEqual(formal, deep_update(cfg, {'dataset': {'cache_in_memory': True}}))
            self.assertTrue(canonical(dataset)['dataset']['cache_in_memory'])

    def test_layer_ownership_and_dataset_separation(self):
        defaults = load_yaml(ROOT / 'configs/default.yaml')
        for key in ('name', 'data_dir', 'split'):
            self.assertNotIn(key, defaults['dataset'])
        self.assertEqual({p.name for p in (ROOT / 'configs/ablations').iterdir()}, {'issm', 'anuga'})
        for dataset in ('issm', 'anuga'):
            self.assertEqual({p.name for p in (ROOT / f'configs/ablations/{dataset}').iterdir()}, {
                'history', 'architecture', 'training_horizon', 'rollout_start', 'temporal_consistency'})
        for name in ('issm', 'anuga', 'adcirc'):
            cfg = load_yaml(ROOT / f'configs/datasets/{name}.yaml')
            self.assertEqual(set(cfg), {'dataset'})
            self.assertNotIn('num_workers', cfg['dataset'])
            self.assertNotIn('train_series_per_scenario_per_epoch', cfg['dataset'])
        self.assertEqual(load_yaml(ROOT / 'configs/datasets/adcirc.yaml')['dataset']['split'], {
            'strategy': 'random', 'train': .6, 'val': .2, 'test': .2, 'seed': 42})

    def test_retired_controls_stay_out_of_active_sources(self):
        removed = ('window' + '_reference', 'full_rollout' + '_known_steps', 'full_rollout' + '_on_val',
                   'windows' + '_per_scenario', 'epoch_num' + '_windows')
        for directory in ('configs', 'datasets', 'training', 'scripts', 'utils'):
            for path in (ROOT / directory).rglob('*'):
                if path.suffix not in ('.py', '.yaml'):
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    source = path.read_text()
                    for key in removed:
                        self.assertNotIn(key, source)

    def test_tc_modes_preserve_corrected_parameters(self):
        modes = ('none', 'adjacent_increment', 'random_pair_increment', 'multiscale_rate', 'rate_curvature', 'hybrid')
        for dataset in ('issm', 'anuga'):
            for index, mode in enumerate(modes):
                cfg = load_yaml(ROOT / f'configs/ablations/{dataset}/temporal_consistency/tc{index}.yaml')['training']['temporal_consistency']
                self.assertEqual(cfg['mode'], mode)
                self.assertEqual(cfg['enabled'], index != 0)
                self.assertEqual(cfg['weight'], 1.)
                if index in (3, 4):
                    self.assertEqual(cfg['rate']['lags'], [1, 3, 6, 12])
                if index == 5:
                    self.assertEqual(cfg['rate']['lags'], [3, 6, 12])


if __name__ == '__main__':
    unittest.main()
