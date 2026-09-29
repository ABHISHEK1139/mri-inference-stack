"""Project readiness preflight checks for local runs, demos, and CI."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    details: str
    required: bool


def _run_command(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception as exc:
        return False, str(exc)

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    message = stdout if stdout else stderr
    if completed.returncode == 0:
        return True, message or "ok"
    return False, message or f"return code {completed.returncode}"


def _check_exists(path: Path, name: str, required: bool) -> CheckResult:
    if path.exists():
        return CheckResult(name=name, status=STATUS_PASS, details=f"found: {path}",
            required=required)
    status = STATUS_FAIL if required else STATUS_WARN
    return CheckResult(name=name, status=status, details=f"missing: {path}", required=required)


def _check_not_lfs_pointer(path: Path, name: str, required: bool) -> CheckResult:
    """Check that a file exists AND is not a Git LFS pointer stub."""
    if not path.exists():
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(name=name, status=status, details=f"missing: {path}", required=required)
    try:
        with path.open("rb") as f:
            header = f.read(48)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            # Honour the caller's `required` flag. Hard-coding required=True made
            # an optional LFS stub fail the whole preflight, which is exactly
            # what `--require-weights` is meant to control.
            status = STATUS_FAIL if required else STATUS_WARN
            return CheckResult(
                name=name,
                status=status,
                details=f"Git LFS pointer (not actual model): {path}. Run `git lfs pull`.",
                required=required,
            )
        size = path.stat().st_size
        return CheckResult(
            name=name,
            status=STATUS_PASS,
            details=f"valid file: {path} ({size:,} bytes)",
            required=required,
        )
    except OSError as exc:
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(name=name, status=status, details=f"read error: {exc}",
            required=required)


def _check_python_version(min_version: tuple[int, int]) -> CheckResult:
    if sys.version_info >= min_version:
        return CheckResult(
            name="python-version",
            status=STATUS_PASS,
            details=f"{platform.python_version()} >= {min_version[0]}.{min_version[1]}",
            required=True,
        )
    return CheckResult(
        name="python-version",
        status=STATUS_FAIL,
        details=f"{platform.python_version()} < {min_version[0]}.{min_version[1]}",
        required=True,
    )


def _check_detection_config(path: Path, required: bool) -> CheckResult:
    if not path.exists():
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(
            name="detection-threshold-config",
            status=status,
            details=f"missing: {path}",
            required=required,
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        # A missing "threshold" key must not silently pass with the 0.5 default:
        # the app would then use an uncalibrated cutoff, materially changing
        # precision/recall without any visible error.
        if "threshold" not in data:
            return CheckResult(
                name="detection-threshold-config",
                status=STATUS_FAIL if required else STATUS_WARN,
                details=f"missing 'threshold' key in {path}",
                required=required,
            )
        threshold = float(data["threshold"])
        in_range = 0.0 <= threshold <= 1.0
        if not in_range:
            return CheckResult(
                name="detection-threshold-config",
                status=STATUS_FAIL if required else STATUS_WARN,
                details=f"threshold out of range: {threshold}",
                required=required,
            )
        return CheckResult(
            name="detection-threshold-config",
            status=STATUS_PASS,
            details=f"threshold={threshold:.4f}",
            required=required,
        )
    except Exception as exc:
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(
            name="detection-threshold-config",
            status=status,
            details=f"invalid json: {exc}",
            required=required,
        )


def _check_command_available(command: str, version_args: Iterable[str],
    required: bool) -> CheckResult:
    command_path = shutil.which(command)
    if command_path is None:
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(
            name=f"command-{command}",
            status=status,
            details="not found in PATH",
            required=required,
        )

    ok, detail = _run_command([command, *version_args])
    status = STATUS_PASS if ok else (STATUS_FAIL if required else STATUS_WARN)
    return CheckResult(
        name=f"command-{command}",
        status=status,
        details=detail,
        required=required,
    )


def _check_command_available_variants(
    command: str,
    version_args_variants: Iterable[Iterable[str]],
    required: bool,
) -> CheckResult:
    command_path = shutil.which(command)
    if command_path is None:
        status = STATUS_FAIL if required else STATUS_WARN
        return CheckResult(
            name=f"command-{command}",
            status=status,
            details="not found in PATH",
            required=required,
        )

    last_detail = "unknown error"
    for version_args in version_args_variants:
        ok, detail = _run_command([command, *version_args])
        if ok:
            return CheckResult(
                name=f"command-{command}",
                status=STATUS_PASS,
                details=detail,
                required=required,
            )
        last_detail = detail

    status = STATUS_FAIL if required else STATUS_WARN
    return CheckResult(
        name=f"command-{command}",
        status=status,
        details=last_detail,
        required=required,
    )


def run_preflight(args: argparse.Namespace) -> list[CheckResult]:
    root = Path(__file__).resolve().parent.parent
    # Use getattr so a programmatically-constructed partial Namespace works.
    require_weights = getattr(args, "require_weights", False)
    require_datasets = getattr(args, "require_datasets", False)
    ci_mode = getattr(args, "ci_mode", False)

    required_files = [
        root / "app.py",
        root / "train.py",
        root / "config.py",
        root / "preprocessing.py",
        root / "requirements.txt",
        root / "requirements.lock",
        root / "pyproject.toml",
        root / "README.md",
        root / "Dockerfile",
        root / "docker-compose.yml",
        root / "k8s" / "deployment.yaml",
        root / "k8s" / "service.yaml",
        root / "k8s" / "networkpolicy.yaml",
        root / "k8s" / "pdb.yaml",
        root / "ansible" / "site.yml",
    ]

    # The training engine is a package; a missing module means a broken import
    # graph that only surfaces at run time.
    required_modules = [
        root / "training" / "runtime.py",
        root / "training" / "state.py",
        root / "training" / "reproducibility.py",
        root / "training" / "data_sources.py",
        root / "training" / "callbacks.py",
        root / "training" / "tracks" / "detection.py",
        root / "training" / "tracks" / "classifier.py",
        root / "training" / "tracks" / "segmentation.py",
        root / "training" / "tracks" / "gan.py",
        root / "training" / "tracks" / "gan_v2.py",
        root / "training" / "tracks" / "gan_augmented.py",
    ]

    weight_files = [
        root / "weights" / "detection_model.keras",
        root / "weights" / "classifier_model.keras",
    ]

    # Matches pyproject's requires-python and scripts/preflight.py's own floor.
    min_python = (3, 10)

    checks: list[CheckResult] = []
    checks.append(_check_python_version(min_python))

    for path in required_files + required_modules:
        checks.append(_check_exists(path, name=f"file-{path.name}", required=True))

    for path in weight_files:
        checks.append(
            _check_not_lfs_pointer(
                path,
                name=f"artifact-{path.name}",
                required=require_weights,
            )
        )

    checks.append(
        _check_detection_config(
            root / "weights" / "detection_inference_config.json",
            required=require_weights,
        )
    )

    if require_datasets:
        checks.append(
            _check_exists(root / "data" / "raw" / "figshare", name="dataset-figshare",
                required=True)
        )
        # BraTS is the segmentation track's only source; checking it here stops
        # preflight passing for a run that will skip track 2.
        checks.append(
            _check_exists(root / "data" / "raw" / "brats", name="dataset-brats", required=False)
        )

    if not ci_mode:
        checks.append(_check_command_available("git", ["--version"], required=False))
        checks.append(_check_command_available("docker", ["--version"], required=False))
        checks.append(
            _check_command_available_variants(
                "kubectl",
                [
                    ["version", "--client", "--short"],
                    ["version", "--client"],
                    ["version"],
                ],
                required=False,
            )
        )

    return checks


def summarize_results(results: list[CheckResult]) -> tuple[bool, str]:
    required_failures = [item for item in results if item.required and item.status == STATUS_FAIL]
    optional_warnings = [item for item in results if item.status == STATUS_WARN]

    lines = ["Preflight readiness report", "=" * 26]
    for item in results:
        lines.append(f"[{item.status}] {item.name}: {item.details}")

    lines.append("")
    lines.append(f"Required failures: {len(required_failures)}")
    lines.append(f"Warnings: {len(optional_warnings)}")

    passed = len(required_failures) == 0
    if passed:
        lines.append("Result: PASS")
    else:
        lines.append("Result: FAIL")

    return passed, "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run project readiness checks")
    parser.add_argument("--ci-mode", action="store_true",
        help="Skip external command checks for CI")
    parser.add_argument("--require-weights", action="store_true",
        help="Fail if core model files are missing")
    parser.add_argument("--require-datasets", action="store_true",
        help="Fail if raw dataset folders are missing")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    results = run_preflight(args)

    if args.json:
        payload = {
            "results": [asdict(item) for item in results],
            "passed": all(not item.required or item.status != STATUS_FAIL for item in results),
        }
        print(json.dumps(payload, indent=2))
    else:
        _, report = summarize_results(results)
        print(report)

    failed = any(item.required and item.status == STATUS_FAIL for item in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
