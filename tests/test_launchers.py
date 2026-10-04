"""Exercise real launch commands without training, tmux, or scheduler side effects."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.io import load_config_bundle

class LauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cogent launcher test ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ('configs', 'scripts', 'shell_scripts_sigspatial_issm', 'sbatch_scripts_sigspatial_issm', 'sbatch_scripts_cercat_anuga'):
            (self.root / directory).symlink_to(ROOT / directory, target_is_directory=True)
        for path in ROOT.glob('*.sh'):
            shutil.copy2(path, self.root / path.name)
        self.capture = self.root / 'commands.jsonl'
        bindir = self.root / 'bin'
        bindir.mkdir()
        fake_python = bindir / 'python'
        fake_python.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
            "with open(os.environ['LAUNCHER_CAPTURE'], 'a') as handle:\n"
            "    handle.write(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in "
            "['HISTORY_LEN','ENCODER','RESIDUAL','ODE_CONTEXT','RELATIVE_TIME','FUTURE_LEN','TC_MODE']}}) + '\\n')\n")
        fake_python.chmod(0o755)
        for command in ('tmux', 'sbatch', 'srun'):
            shutil.copy2(fake_python, bindir / command)
        self.env = dict(os.environ)
        for name in ('DATASET_CONFIG','PROTOCOL_CONFIG','MODEL_CONFIG','RUNTIME_CONFIG','EXTRA_CONFIGS',
                     'TRAIN_ARGS','SLURM_JOB_ID','HISTORY_LEN','FUTURE_LEN','ENCODER','RESIDUAL','ODE_CONTEXT',
                     'RELATIVE_TIME','TC_MODE','TMUX','SESSION_NAME'):
            self.env.pop(name, None)
        self.env.update(PATH=f"{bindir}{os.pathsep}{self.env['PATH']}", PROJECT_ROOT=str(self.root),
            PYTHON_BIN=str(fake_python), LAUNCHER_CAPTURE=str(self.capture), NPROC='1', CUDA_VISIBLE_DEVICES='',
            RUN_NAME='launcher_test', LOG_DIR=str(self.root / 'logs'), LOG_FILE=str(self.root / 'logs/launcher.log'),
            CGE_AUTO_TMUX='0', SUBMIT_SLEEP_SECONDS='0')
        self.selected = dict(HISTORY_LEN='3', ENCODER='lstm', RESIDUAL='off', ODE_CONTEXT='on', RELATIVE_TIME='off')

    def run_script(self, script, *args, success=True, **env):
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(['bash', str(self.root / script), *args], env={**self.env, **env}, cwd=self.root,
                                text=True, capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    @staticmethod
    def train_calls(calls):
        return [call for call in calls if 'scripts/train.py' in call['args']]

    @staticmethod
    def configs(call):
        args = call['args']
        return [args[i + 1] for i, arg in enumerate(args) if arg == '--config']

    def merged(self, call):
        return load_config_bundle([self.root / path for path in self.configs(call)])

    def test_standard_stack_order(self):
        extra = 'configs/ablations/history/h6.yaml'
        for dataset in ('issm', 'anuga'):
            for suffix in ('', '_node2'):
                with self.subTest(dataset=dataset, suffix=suffix):
                    calls = self.train_calls(self.run_script(f'train_{dataset}{suffix}.sh', '--inside-tmux', EXTRA_CONFIGS=extra))
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(self.configs(calls[0]), ['configs/default.yaml', f'configs/datasets/{dataset}.yaml',
                        f'configs/protocols/{dataset}/main.yaml', 'configs/models/node2.yaml', extra, 'configs/runtime/fast.yaml'])
                    self.assertEqual(self.merged(calls[0])['dataset']['history_len'], 6)

    def test_runtime_can_be_disabled(self):
        call = self.train_calls(self.run_script('train_issm.sh', '--inside-tmux', RUNTIME_CONFIG=''))[0]
        self.assertEqual(self.configs(call)[-1], 'configs/models/node2.yaml')

    def test_history_helpers_explicit_controls(self):
        paths = ('shell_scripts_sigspatial_issm/_run_issm_node2_history_scan_local.sh',
                 'sbatch_scripts_sigspatial_issm/_run_issm_node2_history_scan.sh',
                 'sbatch_scripts_cercat_anuga/_run_anuga_node2_history_scan.sh')
        for path in paths:
            with self.subTest(path=path):
                anuga = 'anuga' in path
                call = self.train_calls(self.run_script(path, HISTORY_LEN='6'))[0]
                self.assertEqual(self.configs(call)[4:], ['configs/ablations/history/h6.yaml',
                    f"configs/ablations/future_len/k{64 if anuga else 180}.yaml",
                    f"configs/ablations/rollout_start/known{8 if anuga else 60}.yaml",
                    'configs/ablations/architecture/encoder/transformer.yaml', 'configs/ablations/architecture/residual/on.yaml',
                    'configs/ablations/architecture/ode_context/on.yaml', 'configs/ablations/architecture/relative_time/on.yaml',
                    'configs/ablations/temporal_consistency/tc0_off.yaml', 'configs/runtime/fast.yaml'])

    def test_history_scan_one_through_eight(self):
        calls = self.train_calls(self.run_script('shell_scripts_sigspatial_issm/run_issm_history_scan_sequential.sh'))
        self.assertEqual([self.merged(c)['dataset']['history_len'] for c in calls], list(range(1, 9)))

    def test_architecture_requires_selected_history(self):
        for path in ('shell_scripts_sigspatial_issm/run_issm_architecture_scan_sequential.sh',
                     'sbatch_scripts_sigspatial_issm/submit_issm_architecture_scan.sh'):
            self.assertEqual(self.run_script(path, success=False), [])

    def test_architecture_factorial(self):
        calls = self.train_calls(self.run_script('shell_scripts_sigspatial_issm/run_issm_architecture_scan_sequential.sh', HISTORY_LEN='3'))
        self.assertEqual(len(calls), 16)
        axes = set()
        for call in calls:
            cfg = self.merged(call)
            model = cfg['model']
            axes.add((model['history_encoder']['history_encoder_type'], model['use_residual_decoder'],
                      model['use_history_in_ode'], model['use_relative_time']))
            self.assertEqual(cfg['dataset']['history_len'], 3)
            self.assertEqual(cfg['dataset']['future_len'], 180)
            self.assertEqual(cfg['evaluation']['known_steps'], 60)
            self.assertFalse(cfg['training']['temporal_consistency']['enabled'])
        self.assertEqual(len(axes), 16)

    def test_slurm_architecture_factorial(self):
        calls = self.run_script('sbatch_scripts_sigspatial_issm/submit_issm_architecture_scan.sh', HISTORY_LEN='2')
        self.assertEqual(len(calls), 16)
        self.assertEqual({c['env']['HISTORY_LEN'] for c in calls}, {'2'})
        self.assertEqual(len({tuple(c['env'][k] for k in ('ENCODER','RESIDUAL','ODE_CONTEXT','RELATIVE_TIME')) for c in calls}), 16)

    def test_future_scan_selected_model(self):
        path = 'shell_scripts_sigspatial_issm/run_issm_future_len_ablation_sequential.sh'
        self.assertEqual(self.run_script(path, HISTORY_LEN='3', success=False), [])
        calls = self.train_calls(self.run_script(path, **self.selected))
        self.assertEqual([self.merged(c)['dataset']['future_len'] for c in calls], [30,45,60,75,90,120,150,180])
        for call in calls:
            cfg = self.merged(call)
            self.assertEqual(cfg['dataset']['history_len'], 3)
            self.assertEqual(cfg['model']['history_encoder']['history_encoder_type'], 'lstm')
            self.assertEqual(cfg['model']['relative_time_scale'], 180.)
            self.assertEqual(cfg['evaluation']['known_steps'], 60)
            self.assertFalse(cfg['training']['temporal_consistency']['enabled'])

    def test_tc_six_runs_with_selected_model(self):
        path = 'shell_scripts_sigspatial_issm/run_issm_temporal_consistency_scan_sequential.sh'
        self.assertEqual(self.run_script(path, success=False, **self.selected), [])
        calls = self.train_calls(self.run_script(path, FUTURE_LEN='60', **self.selected))
        self.assertEqual(len(calls), 6)
        self.assertEqual({self.merged(c)['dataset']['future_len'] for c in calls}, {60})
        self.assertEqual({self.merged(c)['training']['temporal_consistency']['mode'] for c in calls},
                         {'none','adjacent_increment','random_pair_increment','multiscale_rate','rate_curvature','hybrid'})

    def test_rollout_start_sweep_one_checkpoint_without_training(self):
        checkpoint = self.root / 'best.pt'
        checkpoint.touch()
        calls = self.run_script('eval_rollout_start_sweep.sh', '--checkpoint', str(checkpoint), '--device', 'cpu')
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.train_calls(calls), [])
        starts = []
        for call in calls:
            args = call['args']
            self.assertIn('scripts/evaluate.py', args)
            self.assertEqual(args[args.index('--checkpoint') + 1], str(checkpoint))
            starts.append(int(args[args.index('--known-steps') + 1]))
        self.assertEqual(starts, [60,90,120])

if __name__ == '__main__':
    unittest.main()
