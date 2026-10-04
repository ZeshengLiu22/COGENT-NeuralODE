"""Run every standalone formal command against a fake Python, including Slurm spooling."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.io import load_config_bundle

PHASE_COUNTS = {'01_history': 8, '02_architecture': 16, '03_training_horizon': 8, '04_temporal_consistency': 6}
CONFIG_FIELDS = ('default', 'dataset', 'protocol', 'model', 'history', 'architecture',
                 'training_horizon', 'rollout_start', 'temporal_consistency', 'runtime')


class LauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cogent launcher test ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ('configs', 'scripts'):
            (self.root / directory).symlink_to(ROOT / directory, target_is_directory=True)
        for path in ROOT.glob('*.sh'):
            shutil.copy2(path, self.root / path.name)
        self.capture = self.root / 'commands.jsonl'
        self.python = self.root / 'python'
        self.python.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
            "with open(os.environ['LAUNCHER_CAPTURE'], 'a') as handle:\n"
            "    handle.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n")
        self.python.chmod(0o755)
        self.env = dict(os.environ)
        for name in ('DATASET_CONFIG', 'PROTOCOL_CONFIG', 'MODEL_CONFIG', 'RUNTIME_CONFIG', 'HISTORY_CONFIG',
                     'ARCHITECTURE_CONFIG', 'TRAINING_HORIZON_CONFIG', 'ROLLOUT_START_CONFIG',
                     'TEMPORAL_CONSISTENCY_CONFIG', 'SLURM_JOB_ID', 'TMUX', 'SESSION_NAME'):
            self.env.pop(name, None)
        self.env.update(PROJECT_ROOT=str(self.root), PYTHON_BIN=str(self.python),
                        LAUNCHER_CAPTURE=str(self.capture), NPROC='4', CUDA_VISIBLE_DEVICES='',
                        RUN_STAMP='test', RUN_NAME='ad_hoc_test',
                        LOG_FILE=str(self.root / 'logs/launcher.log'))

    def run_script(self, path, *args, success=True, **env):
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(['bash', str(path), *args], env={**self.env, **env}, cwd=self.root,
                                text=True, capture_output=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []
        return result, calls

    @staticmethod
    def configs(call):
        args = call['args']
        return [args[i + 1] for i, arg in enumerate(args) if arg == '--config']

    def spooled_script(self, path, select_history=True):
        source = path.read_text()
        dataset = path.relative_to(ROOT / 'launchers').parts[1]
        if select_history:
            source = source.replace('HISTORY_CONFIG="__SET_SELECTED_HISTORY_AFTER_PHASE1__"',
                                    f'HISTORY_CONFIG="configs/ablations/{dataset}/history/h3.yaml"')
        # Slurm uses an explicit environment root. Relocate only that root for this test.
        source = source.replace('PROJECT_ROOT="/home1/09575/zeshengliu/scratch/COGENT-NeuralODE"',
                                f'PROJECT_ROOT="{self.root}"')
        source = source.replace('PROJECT_ROOT="/scratch/09575/zeshengliu/COGENT-NeuralODE"',
                                f'PROJECT_ROOT="{self.root}"')
        spool = self.root / 'scheduler spool'
        spool.mkdir(exist_ok=True)
        target = spool / 'slurm_script'
        target.write_text(source)
        return target

    def test_exact_four_launcher_trees_and_phase_counts(self):
        self.assertEqual(len(list((ROOT / 'launchers').rglob('*.sh'))), 152)
        for execution in ('shell', 'slurm'):
            for dataset in ('issm', 'anuga'):
                base = ROOT / 'launchers' / execution / dataset
                self.assertEqual({p.name for p in base.iterdir()}, set(PHASE_COUNTS))
                for phase, count in PHASE_COUNTS.items():
                    self.assertEqual(len(list((base / phase).glob('*.sh'))), count)

    def test_all_formal_commands_and_independent_phase_controls(self):
        for path in sorted((ROOT / 'launchers').rglob('*.sh')):
            execution, dataset, phase, _ = path.relative_to(ROOT / 'launchers').parts
            with self.subTest(script=path.relative_to(ROOT)):
                result, calls = self.run_script(self.spooled_script(path))
                training = [call for call in calls if 'scripts/train.py' in call['args']]
                self.assertEqual(len(training), 1)
                call = training[0]
                self.assertEqual(call['cwd'], str(self.root))
                self.assertIn('--nproc_per_node=4', call['args'])
                h = int(path.stem[1:]) if phase == '01_history' else 3
                k = int(path.stem[1:]) if phase == '03_training_horizon' else (180 if dataset == 'issm' else 64)
                start = 60 if dataset == 'issm' else 8
                architecture = path.stem if phase == '02_architecture' else 'full'
                tc = path.stem if phase == '04_temporal_consistency' else 'tc0'
                prefix = f'configs/ablations/{dataset}'
                expected = ['configs/default.yaml', f'configs/datasets/{dataset}.yaml',
                    f'configs/protocols/{dataset}/main.yaml', 'configs/models/node2.yaml',
                    f'{prefix}/history/h{h}.yaml', f'{prefix}/architecture/{architecture}.yaml',
                    f'{prefix}/training_horizon/k{k}.yaml', f'{prefix}/rollout_start/known{start}.yaml',
                    f'{prefix}/temporal_consistency/{tc}.yaml', f'configs/runtime/{dataset}_fast.yaml']
                self.assertEqual(self.configs(call), expected)
                cfg = load_config_bundle([self.root / item for item in expected])
                self.assertEqual(cfg['dataset']['history_len'], h)
                self.assertEqual(cfg['dataset']['future_len'], k)
                self.assertEqual(cfg['evaluation']['known_steps'], start)
                self.assertEqual(cfg['dataset']['train_series_per_scenario_per_epoch'], 60 if dataset == 'issm' else 9)
                self.assertEqual(cfg['model']['relative_time_scale'], 180. if dataset == 'issm' else 65.)
                self.assertTrue(cfg['dataset']['cache_in_memory'])
                if phase != '02_architecture':
                    self.assertEqual(cfg['model']['history_encoder']['history_encoder_type'], 'transformer')
                    self.assertTrue(all(cfg['model'][key] for key in (
                        'use_residual_decoder', 'use_history_in_ode', 'use_relative_time')))
                if phase != '04_temporal_consistency':
                    self.assertFalse(cfg['training']['temporal_consistency']['enabled'])
                for field, config in zip(CONFIG_FIELDS, expected):
                    self.assertIn(f'{field}_config={config}', result.stdout)
                self.assertIn('output_dir=', result.stdout)
                self.assertIn('log_file=', result.stdout)

    def test_phases_two_through_four_fail_before_environment_or_training(self):
        for path in sorted((ROOT / 'launchers').rglob('*.sh')):
            if path.parent.name == '01_history':
                continue
            with self.subTest(script=path.relative_to(ROOT)):
                result, calls = self.run_script(path, success=False,
                    HISTORY_CONFIG='configs/ablations/issm/history/h1.yaml')
                self.assertEqual(calls, [])
                self.assertIn('selected Phase-1 history YAML', result.stderr)

    def test_formal_scripts_are_standalone_and_slurm_roots_are_explicit(self):
        for path in (ROOT / 'launchers').rglob('*.sh'):
            source = path.read_text()
            with self.subTest(script=path.relative_to(ROOT)):
                self.assertIsNone(re.search(r'^\s*(?:for|while|until|source)\s', source, re.MULTILINE))
                self.assertNotIn('--array', source)
                self.assertNotIn('sweep_config', source)
                self.assertNotIn('exec bash', source)
                self.assertNotIn('train_issm.sh', source)
                self.assertNotIn('train_anuga.sh', source)
                self.assertIn('scripts/train.py', source)
                if 'slurm' in path.parts:
                    self.assertNotIn('BASH_SOURCE', source)
                    self.assertNotIn('$0', source)
                    self.assertNotIn('dirname', source)
                    self.assertRegex(source, r'PROJECT_ROOT="/(?:home1|scratch)/')
                    self.assertIn('#SBATCH --ntasks-per-node=4', source)
                    self.assertIn('#SBATCH --cpus-per-task=24', source)
                    self.assertIn('#SBATCH -p h100', source)
                    self.assertIn('NPROC=4', source)
                    self.assertIn('\n#SBATCH -A TG-CIS250588\n', source)
                    if 'anuga' in path.parts:
                        self.assertIn('/work2/09575/', source)
                    else:
                        self.assertIn('/work/09575/', source)

    def test_active_shell_syntax(self):
        # Historical directories deliberately excluded from active checks.
        paths = [*ROOT.glob('*.sh'), *(ROOT / 'launchers').rglob('*.sh'), *(ROOT / 'scripts').rglob('*.sh'), *(ROOT / 'tests').rglob('*.sh'),
                 *(ROOT / 'experiments').rglob('*.sh')]
        for path in paths:
            with self.subTest(script=path.relative_to(ROOT)):
                result = subprocess.run(['bash', '-n', str(path)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_historical_launchers_are_retired(self):
        for directory in ('shell_scripts_sigspatial_issm', 'sbatch_scripts_sigspatial_issm', 'sbatch_scripts_cercat_anuga'):
            self.assertFalse((ROOT / directory).exists())
            self.assertTrue((ROOT / 'legacy-scripts' / directory).is_dir())
        self.assertFalse((ROOT / 'scripts/sweep_config.sh').exists())
        self.assertIn('Do not use these scripts for new formal experiments.',
                      (ROOT / 'legacy-scripts/README.md').read_text())

    def test_ad_hoc_launchers_print_protocol_and_named_overlays(self):
        for dataset in ('issm', 'anuga'):
            history = f'configs/ablations/{dataset}/history/h6.yaml'
            result, calls = self.run_script(self.root / f'train_{dataset}.sh', HISTORY_CONFIG=history)
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.configs(calls[0])[4], history)
            self.assertEqual(self.configs(calls[0])[2], f'configs/protocols/{dataset}/main.yaml')
            for field in CONFIG_FIELDS:
                self.assertIn(f'{field}_config=', result.stdout)
        result, calls = self.run_script(self.root / 'train_issm.sh', RUNTIME_CONFIG='')
        self.assertEqual(len(self.configs(calls[0])), 9)
        self.assertNotIn('runtime_config=', result.stdout)

    def test_rollout_start_robustness_uses_same_checkpoint_without_training(self):
        checkpoint = self.root / 'best.pt'
        checkpoint.touch()
        _, calls = self.run_script(self.root / 'eval_rollout_start_sweep.sh',
                                   '--checkpoint', str(checkpoint), '--device', 'cpu')
        self.assertEqual(len(calls), 3)
        starts = []
        for call in calls:
            args = call['args']
            self.assertIn('scripts/evaluate.py', args)
            self.assertNotIn('scripts/train.py', args)
            self.assertEqual(args[args.index('--checkpoint') + 1], str(checkpoint))
            starts.append(int(args[args.index('--known-steps') + 1]))
        self.assertEqual(starts, [60, 90, 120])


if __name__ == '__main__':
    unittest.main()
