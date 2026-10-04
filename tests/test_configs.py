"""Configuration ownership, explicit controls, and recursive merge behavior."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.io import deep_update, load_config_bundle, load_yaml

def canonical(dataset='issm', *overlays):
    return load_config_bundle([ROOT / 'configs' / path for path in ('default.yaml', f'datasets/{dataset}.yaml',
        f'protocols/{dataset}/main.yaml', 'models/node2.yaml', *overlays, 'runtime/fast.yaml')])

class ConfigTest(unittest.TestCase):
    def test_recursive_merge_replaces_lists_preserves_siblings_and_inputs(self):
        base = {'section': {'list': [1,2], 'keep': 3, 'nested': {'x': 1, 'y': 2}}}
        original = deepcopy(base)
        merged = deep_update(base, {'section': {'list': [9], 'nested': {'x': 4}}})
        self.assertEqual(merged, {'section': {'list': [9], 'keep': 3, 'nested': {'x': 4, 'y': 2}}})
        self.assertEqual(base, original)

    def test_left_to_right_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            layers = ('default','dataset','protocol','model','ablation','runtime')
            for index, name in enumerate(layers):
                path = Path(tmp) / f'{name}.yaml'
                path.write_text(f'section:\n  value: {index}\n  {name}: true\n  list: [{index}]\n')
                paths.append(path)
            merged = load_config_bundle(paths)['section']
            self.assertEqual(merged['value'], 5)
            self.assertEqual(merged['list'], [5])
            self.assertTrue(all(merged[name] for name in layers))

    def test_canonical_issm(self):
        cfg = canonical()
        self.assertEqual(cfg['dataset']['data_dir'], './data/ISSM/PIG_5000')
        self.assertEqual((cfg['dataset']['history_len'], cfg['dataset']['future_len'], cfg['evaluation']['known_steps'],
                          cfg['model']['relative_time_scale']), (1,180,60,180.))
        self.assertEqual(cfg['model']['latent_dim'], 96)
        self.assertEqual(cfg['model']['history_encoder']['history_encoder_type'], 'transformer')
        for key in ('use_residual_decoder','use_history_in_ode','use_relative_time'):
            self.assertTrue(cfg['model'][key])
        self.assertEqual(cfg['training']['batch_size'], 8)
        self.assertEqual(cfg['training']['train_horizon_min'], 24)
        self.assertEqual(cfg['training']['loss_scale_factor'], 100.)
        self.assertIsNone(cfg['training']['train_horizon_max'])
        self.assertFalse(cfg['training']['temporal_consistency']['enabled'])
        self.assertEqual(cfg['dataset']['split'], {'strategy':'issm_rate_modulo','modulo':20,'val_remainder':0,'test_remainder':10})

    def test_anuga_protocol_preserved(self):
        cfg = canonical('anuga')
        self.assertEqual((cfg['dataset']['history_len'], cfg['dataset']['future_len'], cfg['evaluation']['known_steps'],
                          cfg['model']['relative_time_scale']), (1,64,8,65.))
        self.assertEqual(cfg['training']['batch_size'], 1)
        self.assertEqual(cfg['training']['train_horizon_min'], 8)
        self.assertEqual(cfg['training']['loss_scale_factor'], 1.)
        self.assertEqual(cfg['dataset']['file_patterns'], ['sim_*_merged.npz'])
        self.assertEqual(cfg['dataset']['split'], {'strategy':'random','train':.6,'val':.2,'test':.2,'seed':42})

    def test_explicit_controls_equal_canonical(self):
        controls = ('history/h1.yaml','future_len/k180.yaml','rollout_start/known60.yaml',
                    'architecture/encoder/transformer.yaml','architecture/residual/on.yaml','architecture/ode_context/on.yaml',
                    'architecture/relative_time/on.yaml','temporal_consistency/tc0_off.yaml')
        self.assertEqual(canonical(), canonical('issm', *(f'ablations/{path}' for path in controls)))

    def test_temporal_overlays_change_only_one_value(self):
        for directory, values, prefix, section, key in (
            ('history',range(1,9),'h','dataset','history_len'),
            ('future_len',(30,45,60,64,75,90,120,150,180),'k','dataset','future_len'),
            ('rollout_start',(8,60,90,120),'known','evaluation','known_steps')):
            for value in values:
                path = f'ablations/{directory}/{prefix}{value}.yaml'
                with self.subTest(path=path):
                    self.assertEqual(load_yaml(ROOT / 'configs' / path), {section:{key:value}})
                    expected = canonical()
                    expected[section][key] = value
                    self.assertEqual(canonical('issm', path), expected)

    def test_architecture_axes_are_independent(self):
        for encoder in ('transformer','lstm'):
            self.assertEqual(load_yaml(ROOT / f'configs/ablations/architecture/encoder/{encoder}.yaml'),
                             {'model':{'history_encoder':{'history_encoder_type':encoder}}})
        for axis,key in (('residual','use_residual_decoder'),('ode_context','use_history_in_ode'),('relative_time','use_relative_time')):
            for label,value in (('on',True),('off',False)):
                self.assertEqual(load_yaml(ROOT / f'configs/ablations/architecture/{axis}/{label}.yaml'), {'model':{key:value}})

    def test_runtime_only_contains_loader_settings(self):
        cfg = load_yaml(ROOT / 'configs/runtime/fast.yaml')
        self.assertEqual(set(cfg), {'dataset'})
        self.assertLessEqual(set(cfg['dataset']), {'num_workers','pin_memory','prefetch_factor','persistent_workers','cache_in_memory'})
        self.assertFalse((ROOT / 'configs/runtime/default.yaml').exists())

    def test_layer_ownership(self):
        defaults = load_yaml(ROOT / 'configs/default.yaml')
        for key in ('name','data_dir','split'):
            self.assertNotIn(key, defaults['dataset'])
        self.assertEqual(sorted(p.relative_to(ROOT / 'configs/protocols').as_posix() for p in (ROOT / 'configs/protocols').rglob('*.yaml')),
                         ['anuga/main.yaml','issm/main.yaml'])
        for name in ('issm','anuga','adcirc'):
            cfg = load_yaml(ROOT / f'configs/datasets/{name}.yaml')
            self.assertEqual(set(cfg), {'dataset'})
            self.assertNotIn('num_workers', cfg['dataset'])

        self.assertEqual(load_yaml(ROOT / "configs/datasets/adcirc.yaml")["dataset"]["split"],
                         {"strategy": "random", "train": 0.6, "val": 0.2, "test": 0.2, "seed": 42})

    def test_removed_temporal_controls_stay_out_of_active_sources(self):
        removed = ('window' + '_reference', 'full_rollout' + '_known_steps', 'full_rollout' + '_on_val')
        for directory in ('configs', 'datasets', 'training', 'scripts', 'utils'):
            for path in (ROOT / directory).rglob('*'):
                if path.suffix not in ('.py', '.yaml'):
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    source = path.read_text()
                    for key in removed:
                        self.assertNotIn(key, source)

    def test_tc_modes_and_preserved_parameters(self):
        for index,mode in enumerate(('none','adjacent_increment','random_pair_increment','multiscale_rate','rate_curvature','hybrid')):
            name = 'off' if index == 0 else mode
            cfg = load_yaml(ROOT / f'configs/ablations/temporal_consistency/tc{index}_{name}.yaml')['training']['temporal_consistency']
            self.assertEqual(cfg['mode'], mode)
            self.assertEqual(cfg['enabled'], index != 0)
            self.assertEqual(cfg['weight'], 1.)
            if index in (3,4): self.assertEqual(cfg['rate']['lags'], [1,3,6,12])
            if index == 5: self.assertEqual(cfg['rate']['lags'], [3,6,12])

if __name__ == '__main__':
    unittest.main()
