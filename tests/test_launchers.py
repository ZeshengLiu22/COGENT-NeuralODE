"""Run every standalone formal command against a fake Python, including scheduler spooling."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
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
        self.metric_table = (
            'scope          channel    count  phys_rmse  phys_mae  norm_rmse  norm_mae\n'
            'whole_rollout  aggregate  36     1.250000   1.000000  0.500000   0.400000\n'
            'final_step     aggregate  9      2.000000   1.500000  0.800000   0.600000'
        )
        self.python.write_text(f'#!{sys.executable}\nMETRIC_TABLE = {self.metric_table!r}\n' + textwrap.dedent(r"""
            import json, os, sys
            from pathlib import Path
            args = sys.argv[1:]
            with open(os.environ['LAUNCHER_CAPTURE'], 'a') as handle:
                handle.write(json.dumps({'args': args, 'cwd': os.getcwd()}) + '\n')
            def option(name):
                return args[args.index(name) + 1]
            if 'scripts/train.py' in args and option('--run-name').endswith('/train'):
                output = Path('outputs') / option('--run-name')
                output.mkdir(parents=True, exist_ok=True)
                print('fake training stdout')
                print('fake training stderr', file=sys.stderr)
                (output / 'train.log').write_text('retained training log\n')
                (output / 'history.json').write_text('[{"epoch": 1}]\n')
                (output / 'config.json').write_text('{}\n')
                (output / 'split_files.json').write_text('{}\n')
                configs = [args[i + 1] for i, arg in enumerate(args) if arg == '--config']
                (output / 'config_stack.txt').write_text('\n'.join(configs) + '\n')
                checkpoint = os.environ.get('FAKE_CHECKPOINT', 'valid')
                if checkpoint != 'missing':
                    (output / 'best.pt').write_bytes(b'' if checkpoint == 'empty' else b'best validation weights')
                status = int(os.environ.get('FAKE_TRAIN_EXIT', '0'))
                if status:
                    print('fake training failure', file=sys.stderr)
                else:
                    (output / 'final_metrics.json').write_text('{"best_epoch": 1}\n')
                sys.exit(status)
            elif 'scripts/evaluate.py' in args and Path(option('--checkpoint')).parent.name == 'train':
                output = Path(option('--output-dir'))
                output.mkdir(parents=True, exist_ok=True)
                status = int(os.environ.get('FAKE_INFER_EXIT', '0'))
                if status:
                    print('fake inference failure', file=sys.stderr)
                    sys.exit(status)
                print('Detailed metric table:\n' + METRIC_TABLE)
                (output / 'metric_table.txt').write_text(METRIC_TABLE + '\n')
        """))
        self.python.chmod(0o755)
        self.env = dict(os.environ)
        for name in ('DATASET_CONFIG', 'PROTOCOL_CONFIG', 'MODEL_CONFIG', 'RUNTIME_CONFIG', 'HISTORY_CONFIG',
                     'ARCHITECTURE_CONFIG', 'TRAINING_HORIZON_CONFIG', 'ROLLOUT_START_CONFIG',
                     'TEMPORAL_CONSISTENCY_CONFIG', 'SLURM_JOB_ID', 'PBS_JOBID', 'PBS_O_WORKDIR',
                     'SITE_CONFIG', 'FINAL_CONFIG', 'TMUX', 'SESSION_NAME',
                     'FAKE_TRAIN_EXIT', 'FAKE_INFER_EXIT', 'FAKE_CHECKPOINT'):
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
        target = spool / 'scheduler_script'
        target.write_text(source)
        return target

    def test_exact_six_launcher_trees_and_phase_counts(self):
        self.assertEqual(len(list((ROOT / 'launchers').rglob('*.sh'))), 228)
        pbs_names = [re.search(r'^#PBS -N (.+)$', path.read_text(), re.MULTILINE).group(1)
                     for path in (ROOT / 'launchers/PBS').rglob('*.sh')]
        self.assertEqual(len(set(pbs_names)), 76)
        self.assertTrue(all(len(name) <= 15 for name in pbs_names))
        for execution in ('shell', 'slurm', 'PBS'):
            for dataset in ('issm', 'anuga'):
                base = ROOT / 'launchers' / execution / dataset
                self.assertEqual({p.name for p in base.iterdir()}, set(PHASE_COUNTS))
                for phase, count in PHASE_COUNTS.items():
                    self.assertEqual(len(list((base / phase).glob('*.sh'))), count)

    def test_all_formal_commands_and_independent_phase_controls(self):
        for path in sorted((ROOT / 'launchers').rglob('*.sh')):
            execution, dataset, phase, _ = path.relative_to(ROOT / 'launchers').parts
            with self.subTest(script=path.relative_to(ROOT)):
                result, calls = self.run_script(self.spooled_script(path),
                    PBS_JOBID='12345.casper-pbs', PBS_O_WORKDIR=str(self.root / 'unrelated submit directory'))
                training = [call for call in calls if 'scripts/train.py' in call['args']]
                self.assertEqual(len(training), 1)
                call = training[0]
                self.assertEqual(call['cwd'], str(self.root))
                nproc = 1 if execution == 'PBS' else 4
                self.assertIn(f'--nproc_per_node={nproc}', call['args'])
                if execution == 'PBS':
                    self.assertIn('--standalone', call['args'])
                    self.assertIn('pbs_job_id=12345.casper-pbs', result.stdout)
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
                fields = list(CONFIG_FIELDS)
                if execution == 'PBS':
                    if dataset == 'anuga':
                        expected.append('configs/runtime/anuga_casper.yaml')
                        fields.append('site')
                    expected.append('configs/runtime/single_a100.yaml')
                    fields.append('final')
                self.assertEqual(self.configs(call), expected)
                cfg = load_config_bundle([self.root / item for item in expected])
                self.assertEqual(cfg['dataset']['history_len'], h)
                self.assertEqual(cfg['dataset']['future_len'], k)
                self.assertEqual(cfg['evaluation']['known_steps'], start)
                self.assertEqual(cfg['dataset']['train_series_per_scenario_per_epoch'], 60 if dataset == 'issm' else 9)
                self.assertEqual(cfg['model']['relative_time_scale'], 180. if dataset == 'issm' else 65.)
                self.assertTrue(cfg['dataset']['cache_in_memory'])
                self.assertEqual(cfg['amp']['mode'], 'none')
                self.assertEqual(cfg['evaluation']['amp_mode'], 'none')
                self.assertEqual(cfg['training']['grad_accum_steps'], 4 if execution == 'PBS' else 1)
                batch = 8 if dataset == 'issm' else 1
                self.assertEqual(cfg['training']['batch_size'], batch)
                self.assertEqual(nproc * batch * cfg['training']['grad_accum_steps'], 4 * batch)
                self.assertEqual(cfg['training']['epochs'], 300)
                self.assertEqual(cfg['training']['lr'], 0.001)
                self.assertEqual(cfg['training']['scheduler'], 'cosine')
                if execution == 'PBS':
                    data_dir = './data/ISSM/PIG_5000' if dataset == 'issm' else './data/ANUGA/simulation_data_merged'
                    self.assertEqual(cfg['dataset']['data_dir'], data_dir)
                if phase != '02_architecture':
                    self.assertEqual(cfg['model']['history_encoder']['history_encoder_type'], 'transformer')
                    self.assertTrue(all(cfg['model'][key] for key in (
                        'use_residual_decoder', 'use_history_in_ode', 'use_relative_time')))
                if phase != '04_temporal_consistency':
                    self.assertFalse(cfg['training']['temporal_consistency']['enabled'])
                for field, config in zip(fields, expected):
                    self.assertIn(f'{field}_config={config}', result.stdout)
                self.assertIn('output_dir=', result.stdout)
                self.assertIn('log_file=', result.stdout)
                evaluations = [entry for entry in calls if 'scripts/evaluate.py' in entry['args']]
                if execution != 'PBS':
                    self.assertEqual(evaluations, [])
                    continue
                experiment = f'{dataset}_{phase}_h{h}_k{k}_{architecture}_{tc}_test_pbs12345.casper-pbs'
                args = call['args']
                self.assertEqual(args[args.index('--run-name') + 1], f'{experiment}/train')
                run_dir = self.root / 'outputs' / experiment
                train_dir, infer_dir = run_dir / 'train', run_dir / 'inference'
                self.assertEqual(len(evaluations), 1)
                self.assertGreater(calls.index(evaluations[0]), calls.index(call))
                self.assertEqual(evaluations[0]['cwd'], str(self.root))
                self.assertEqual(evaluations[0]['args'], [
                    'scripts/evaluate.py', '--checkpoint', str(train_dir / 'best.pt'),
                    '--split', 'test', '--device', 'cuda', '--amp-mode', 'none',
                    '--output-dir', str(infer_dir)])
                self.assertEqual({entry.name for entry in run_dir.iterdir()}, {'train', 'inference'})
                for name in ('best.pt', 'config.json', 'config_stack.txt', 'split_files.json',
                             'history.json', 'final_metrics.json', 'train.log', 'launcher.log',
                             'runtime_metadata.txt'):
                    self.assertTrue((train_dir / name).is_file(), name)
                self.assertEqual((train_dir / 'config_stack.txt').read_text().splitlines(), expected)
                train_log = (train_dir / 'launcher.log').read_text()
                infer_log = (infer_dir / 'inference.log').read_text()
                self.assertIn('fake training stdout', train_log)
                self.assertIn('fake training stderr', train_log)
                self.assertNotIn('Detailed metric table:', train_log)
                self.assertNotIn('fake training stdout', infer_log)
                self.assertIn('Detailed metric table:\n' + self.metric_table, infer_log)
                self.assertEqual((infer_dir / 'metric_table.txt').read_text(), self.metric_table + '\n')
                metadata = (train_dir / 'runtime_metadata.txt').read_text()
                for key in ('pbs_job_id', 'git_commit', 'gpu_model', 'run_name',
                            'training_start_utc', 'training_end_utc', 'inference_start_utc',
                            'inference_end_utc'):
                    self.assertIn(f'{key}=', metadata)
                for stage in ('training', 'inference'):
                    self.assertIn(f'{stage}_exit_status=0', metadata)
                for field, config in zip(fields, expected):
                    self.assertIn(f'{field}_config={config}', metadata)

    def test_phases_two_through_four_fail_before_environment_or_training(self):
        for path in sorted((ROOT / 'launchers').rglob('*.sh')):
            if path.parent.name == '01_history':
                continue
            with self.subTest(script=path.relative_to(ROOT)):
                result, calls = self.run_script(path, success=False,
                    HISTORY_CONFIG='configs/ablations/issm/history/h1.yaml')
                self.assertEqual(calls, [])
                self.assertIn('selected Phase-1 history YAML', result.stderr)

    def test_formal_scripts_are_standalone_and_scheduler_roots_are_explicit(self):
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
                if 'slurm' in path.parts or 'PBS' in path.parts:
                    self.assertNotIn('BASH_SOURCE', source)
                    self.assertNotIn('$0', source)
                    self.assertNotIn('dirname', source)
                if 'slurm' in path.parts:
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
                elif 'PBS' in path.parts:
                    self.assertTrue(source.startswith('#!/bin/bash -l\n'))
                    self.assertEqual([line for line in source.splitlines() if line.startswith('#PBS')], [
                        f'#PBS -N {path.parts[-3]}_{path.parts[-2][:2]}_{path.stem.split("_")[0]}',
                        '#PBS -A ULHI0006', '#PBS -q casper',
                        '#PBS -l select=1:ncpus=16:mpiprocs=1:mem=128GB:ngpus=1:gpu_type=a100_80gb',
                        '#PBS -l place=shared', '#PBS -l walltime=24:00:00', '#PBS -j oe'])
                    self.assertIn('PROJECT_ROOT="${PROJECT_ROOT:-/glade/u/home/zel/scratch/COGENT-NeuralODE}"', source)
                    self.assertIn('/glade/work/zel/conda-envs/casper-ml/bin/python', source)
                    self.assertIn('\nNPROC=1\n', source)
                    self.assertNotIn('SLURM', source)
                    self.assertNotIn('#SBATCH', source)
                    self.assertTrue(os.access(path, os.X_OK))
                    self.assertIn('set -euo pipefail', source)

    def test_pbs_failures_preserve_artifacts_and_return_nonzero(self):
        cases = (
            ('training', {'FAKE_TRAIN_EXIT': '7'}, 7),
            ('missing', {'FAKE_CHECKPOINT': 'missing'}, 1),
            ('empty', {'FAKE_CHECKPOINT': 'empty'}, 1),
            ('inference', {'FAKE_INFER_EXIT': '9'}, 9),
        )
        for dataset in ('issm', 'anuga'):
            path = ROOT / f'launchers/PBS/{dataset}/01_history/h1.sh'
            for case, env, exit_code in cases:
                with self.subTest(dataset=dataset, case=case):
                    result, calls = self.run_script(self.spooled_script(path), success=False,
                        RUN_STAMP=case, PBS_JOBID='77.casper-pbs', **env)
                    self.assertEqual(result.returncode, exit_code)
                    train = next(call for call in calls if 'scripts/train.py' in call['args'])
                    run_name = train['args'][train['args'].index('--run-name') + 1]
                    train_dir = self.root / 'outputs' / run_name
                    infer_dir = train_dir.parent / 'inference'
                    self.assertTrue((train_dir / 'history.json').is_file())
                    self.assertTrue((train_dir / 'train.log').is_file())
                    metadata = (train_dir / 'runtime_metadata.txt').read_text()
                    self.assertIn('training_end_utc=', metadata)
                    evaluations = [call for call in calls if 'scripts/evaluate.py' in call['args']]
                    if case == 'inference':
                        self.assertEqual(len(evaluations), 1)
                        self.assertIn('fake inference failure', (infer_dir / 'inference.log').read_text())
                        self.assertEqual((train_dir / 'best.pt').read_bytes(), b'best validation weights')
                        self.assertEqual(json.loads((train_dir / 'final_metrics.json').read_text()), {'best_epoch': 1})
                        self.assertIn('inference_end_utc=', metadata)
                        self.assertIn('inference_exit_status=9', metadata)
                    else:
                        self.assertEqual(evaluations, [])
                        self.assertFalse((infer_dir / 'inference.log').exists())
                        self.assertNotIn('inference_start_utc=', metadata)
                        if case == 'training':
                            self.assertIn('training_exit_status=7', metadata)
                            self.assertIn('fake training failure', (train_dir / 'launcher.log').read_text())
                            self.assertEqual((train_dir / 'best.pt').read_bytes(), b'best validation weights')
                        else:
                            self.assertIn('Best checkpoint is missing or empty:', result.stderr)

    def test_pbs_distinct_job_ids_and_atomic_run_directory_reservation(self):
        for dataset in ('issm', 'anuga'):
            path = self.spooled_script(ROOT / f'launchers/PBS/{dataset}/01_history/h1.sh')
            directories = []
            for job_id in ('100.casper-pbs', '101.casper-pbs'):
                _, calls = self.run_script(path, RUN_STAMP='same_timestamp', PBS_JOBID=job_id)
                train = next(call for call in calls if 'scripts/train.py' in call['args'])
                run_name = train['args'][train['args'].index('--run-name') + 1]
                directories.append(self.root / 'outputs' / run_name)
            self.assertNotEqual(*directories)
            _, calls = self.run_script(path, success=False,
                RUN_STAMP='same_timestamp', PBS_JOBID='100.casper-pbs')
            self.assertEqual(calls, [])
            for directory in directories:
                self.assertEqual((directory / 'best.pt').read_bytes(), b'best validation weights')
                self.assertTrue((directory.parent / 'inference/metric_table.txt').exists())

    def test_pbs_optional_metadata_failures_do_not_block_workflow(self):
        unavailable = self.root / 'unavailable tools'
        unavailable.mkdir()
        for name in ('git', 'nvidia-smi'):
            tool = unavailable / name
            tool.write_text('#!/bin/bash\nexit 23\n')
            tool.chmod(0o755)
        for dataset in ('issm', 'anuga'):
            path = self.spooled_script(ROOT / f'launchers/PBS/{dataset}/01_history/h1.sh')
            # A directory cannot be opened as a metadata file, even when running as root.
            path.write_text(path.read_text().replace(
                'METADATA_FILE="$TRAIN_DIR/runtime_metadata.txt"', 'METADATA_FILE="$TRAIN_DIR"'))
            result, calls = self.run_script(path, PBS_JOBID='200.casper-pbs',
                PATH=str(unavailable) + os.pathsep + self.env['PATH'])
            self.assertIn('git_commit=unavailable', result.stdout)
            self.assertIn('gpu_model=unavailable', result.stdout)
            self.assertEqual(sum('scripts/train.py' in call['args'] for call in calls), 1)
            self.assertEqual(sum('scripts/evaluate.py' in call['args'] for call in calls), 1)
            self.assertIn('Detailed metric table:\n' + self.metric_table, result.stdout)

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
