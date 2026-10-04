"""Exercise launcher commands without starting training, tmux, or Slurm jobs."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_config_bundle, load_yaml


TC_CONFIG = "configs/TemporalConsistency/tc3_multiscale_rate.yaml"
ISSM_LOADER = "configs/ISSM_History_Scan/issm_pig5000_fast_loader.yaml"
ANUGA_LOADER = "configs/ANUGA_History_Scan/anuga_fast_loader.yaml"
ABLATION_CONFIG = "configs/NODE2_Upgrade1_Ablation/model_node2_no_relative_time.yaml"
FORMAL_HELPERS = (
    "sbatch_scripts_sigspatial_issm/_run_issm_node2_history_scan.sh",
    "shell_scripts_sigspatial_issm/_run_issm_node2_history_scan_local.sh",
    "sbatch_scripts_sigspatial_issm/_run_issm_node2_future_len_ablation.sh",
    "shell_scripts_sigspatial_issm/_run_issm_node2_future_len_ablation_local.sh",
    "sbatch_scripts_sigspatial_issm/_run_issm_node2_upgrade1_ablation.sh",
    "shell_scripts_sigspatial_issm/_run_issm_node2_upgrade1_ablation_local.sh",
    "sbatch_scripts_cercat_anuga/_run_anuga_node2_history_scan.sh",
)


class LauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cogent launcher test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "configs").symlink_to(PROJECT_ROOT / "configs", target_is_directory=True)
        for name in ("train_issm.sh", "train_issm_node2.sh", "train_anuga.sh", "train_anuga_node2.sh"):
            shutil.copy2(PROJECT_ROOT / name, self.root / name)
        self.capture = self.root / "commands.jsonl"
        self.fake_python = self.root / "fake_python"
        self.fake_python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['LAUNCHER_CAPTURE'], 'a') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        self.fake_python.chmod(0o755)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for command in ("tmux", "sbatch", "srun"):
            blocked = self.bin_dir / command
            blocked.write_text("#!/usr/bin/env bash\necho 'Unexpected scheduler call' >&2\nexit 99\n")
            blocked.chmod(0o755)

    def _launch(self, script: str, **overrides: str) -> tuple[list[str], dict]:
        self.capture.unlink(missing_ok=True)
        env = dict(os.environ)
        for name in (
            "BASE_CONFIG", "DATASET_CONFIG", "MODEL_CONFIG", "LOADER_CONFIG", "EXTRA_CONFIGS",
            "TRAIN_ARGS", "FUTURE_CONFIG_ROOT", "SLURM_JOB_ID", "MASTER_PORT", "RUN_VARIANT",
        ):
            env.pop(name, None)
        env.update({
            "PATH": f"{self.bin_dir}{os.pathsep}{env['PATH']}",
            "PROJECT_ROOT": str(self.root),
            "PYTHON_BIN": str(self.fake_python),
            "LAUNCHER_CAPTURE": str(self.capture),
            "NPROC": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HISTORY_LEN": "6",
            "FUTURE_LEN": "30",
            "ABLATION_NAME": "no_relative_time",
            "ABLATION_CONFIG": ABLATION_CONFIG,
            "RUN_NAME": "launcher_test",
            "LOG_DIR": str(self.root / "logs"),
            "LOG_FILE": str(self.root / "logs" / "launcher.log"),
        })
        env.update(overrides)
        if "/" not in script:
            command = ["bash", str(self.root / script), "--inside-tmux"]
        else:
            command = ["bash", str(PROJECT_ROOT / script)]
        result = subprocess.run(command, env=env, cwd=self.root, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        train_calls = [args for args in calls if "scripts/train.py" in args]
        self.assertEqual(len(train_calls), 1, calls)
        args = train_calls[0]
        configs = [args[index + 1] for index, arg in enumerate(args) if arg == "--config"]
        self.assertNotIn("", configs)
        merged = load_config_bundle([self.root / path for path in configs])
        return configs, merged

    def test_standard_launchers_preserve_default_protocol(self) -> None:
        for dataset in ("issm", "anuga"):
            for suffix in ("", "_node2"):
                with self.subTest(dataset=dataset, wrapper=suffix):
                    configs, merged = self._launch(f"train_{dataset}{suffix}.sh", EXTRA_CONFIGS=TC_CONFIG)
                    self.assertEqual(configs[1:], [f"configs/{dataset}.yaml", "configs/model_node2.yaml", TC_CONFIG])
                    self.assertEqual(merged["model"]["relative_time_scale"], 180.0 if dataset == "issm" else 65.0)
                    self.assertEqual(merged["training"]["temporal_consistency"]["mode"], "multiscale_rate")

    def test_paper_base_automatically_skips_conflicting_dataset_overlay(self) -> None:
        for script in ("train_issm.sh", "train_issm_node2.sh"):
            for explicit_empty in (False, True):
                with self.subTest(script=script, explicit_empty=explicit_empty):
                    overrides = {"DATASET_CONFIG": ""} if explicit_empty else {}
                    configs, merged = self._launch(
                        script, BASE_CONFIG="configs/issm_paper_matched.yaml",
                        LOADER_CONFIG=ISSM_LOADER, EXTRA_CONFIGS=TC_CONFIG, **overrides,
                    )
                    self.assertEqual(configs, [
                        "configs/issm_paper_matched.yaml", "configs/model_node2.yaml", ISSM_LOADER, TC_CONFIG,
                    ])
                    self.assertEqual(merged["dataset"]["history_len"], 1)
                    self.assertEqual(merged["dataset"]["future_len"], 239)
                    self.assertEqual(merged["model"]["relative_time_scale"], 239.0)
                    self.assertEqual(merged["evaluation"]["full_rollout_known_steps"], 1)
                    self.assertEqual(merged["dataset"]["data_dir"], "./data/ISSM/PIG_5000")

    def test_formal_tc_overlay_preserves_loader_and_ablation(self) -> None:
        for script in FORMAL_HELPERS:
            with self.subTest(script=script):
                configs, merged = self._launch(script, EXTRA_CONFIGS=TC_CONFIG)
                dataset = "anuga" if "anuga" in script else "issm"
                loader = ANUGA_LOADER if dataset == "anuga" else ISSM_LOADER
                expected = [f"configs/{dataset}.yaml", "configs/model_node2.yaml"]
                if "upgrade1_ablation" in script:
                    expected.append(ABLATION_CONFIG)
                    self.assertFalse(merged["model"]["use_relative_time"])
                self.assertEqual(configs[1:], expected + [loader, TC_CONFIG])
                self.assertEqual(merged["dataset"]["data_dir"], load_yaml(PROJECT_ROOT / loader)["dataset"]["data_dir"])
                self.assertTrue(merged["dataset"]["persistent_workers"])
                self.assertEqual(merged["training"]["temporal_consistency"]["mode"], "multiscale_rate")

    def test_all_launchers_accept_explicitly_empty_optional_overlays(self) -> None:
        for script in ("train_issm.sh", "train_anuga.sh", *FORMAL_HELPERS):
            with self.subTest(script=script):
                configs, _ = self._launch(script, DATASET_CONFIG="", LOADER_CONFIG="", EXTRA_CONFIGS=TC_CONFIG)
                expected = ["configs/model_node2.yaml"]
                if "upgrade1_ablation" in script:
                    expected.append(ABLATION_CONFIG)
                self.assertEqual(configs[1:], expected + [TC_CONFIG])

    def test_history_and_future_scans_change_only_scanned_length(self) -> None:
        for directory, varying in (
            ("ISSM_History_Scan", "history_len"),
            ("ANUGA_History_Scan", "history_len"),
            ("ISSM_Future_Len_Ablation", "future_len"),
        ):
            reference = None
            configs = sorted((PROJECT_ROOT / "configs" / directory).glob("base_*.yaml"))
            self.assertEqual(len(configs), 8)
            for path in configs:
                with self.subTest(config=path.name):
                    config = deepcopy(load_yaml(path))
                    config["dataset"].pop(varying)
                    if reference is None:
                        reference = config
                    self.assertEqual(config, reference)
                    self.assertEqual(config["training"]["grad_clip_norm_dtype"], "fp32")


if __name__ == "__main__":
    unittest.main()
