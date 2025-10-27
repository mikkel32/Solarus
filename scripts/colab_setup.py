from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

COLAB_ROOT = Path("/content")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_NAME = "solarus_workspace"
DEFAULT_MODELS_SUBDIR = "models"
DATA_DIR_NAME = "data"


def running_in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or COLAB_ROOT.exists()


def ensure_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: unable to create directory {path}: {exc}")
        return False
    return True


def copy_datasets(source_dir: Path, target_dir: Path) -> List[Path]:
    copied: List[Path] = []
    if not source_dir.exists():
        print(f"Warning: dataset directory {source_dir} does not exist; skipping copy.")
        return copied
    if not ensure_directory(target_dir):
        return copied
    for item in source_dir.glob("*.csv"):
        destination = target_dir / item.name
        try:
            if destination.exists():
                continue
        except OSError:
            continue
        try:
            shutil.copy2(item, destination)
        except OSError as exc:
            print(f"Warning: failed to copy {item} to {destination}: {exc}")
        else:
            copied.append(destination)
            print(f"Copied {item.name} to {destination}")
    return copied


def install_packages(packages: Sequence[str], *, quiet: bool = True) -> List[str]:
    installed: List[str] = []
    if not packages:
        return installed
    for spec in packages:
        command = [sys.executable, "-m", "pip", "install"]
        if quiet:
            command.append("--quiet")
        command.append(spec)
        print(f"Installing {spec} via pip ...")
        result = subprocess.run(command, check=False, capture_output=quiet)
        if result.returncode != 0:
            print(f"Warning: pip failed with exit code {result.returncode} while installing {spec}.")
            if quiet and result.stderr:
                print(result.stderr.decode("utf-8", errors="ignore"))
        else:
            installed.append(spec)
    return installed


def extras_packages(selection: str) -> List[str]:
    mapping = {
        "none": [],
        "transformers": ["transformers>=4.34", "sentence-transformers>=2.2.2"],
        "ui": ["gradio>=4.0", "transformers>=4.34", "sentence-transformers>=2.2.2"],
        "all": ["transformers>=4.34", "sentence-transformers>=2.2.2", "gradio>=4.0"],
    }
    return mapping.get(selection, [])


def normalise_workspace(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser()
    return COLAB_ROOT / DEFAULT_WORKSPACE_NAME


def stage_workspace(workspace_root: Path, models_subdir: str) -> tuple[Path, Path]:
    ensure_directory(workspace_root)
    models_dir = workspace_root / models_subdir
    ensure_directory(models_dir)
    return workspace_root, models_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Google Colab environment for Solarus training by staging datasets, "
            "creating workspace directories, and installing optional dependencies."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "Target workspace root on Colab (defaults to /content/solarus_workspace). "
            "The directory will be created if it does not exist."
        ),
    )
    parser.add_argument(
        "--models-subdir",
        type=str,
        default=DEFAULT_MODELS_SUBDIR,
        help="Directory name under the workspace root where training artefacts will be stored.",
    )
    parser.add_argument(
        "--no-copy-data",
        dest="copy_data",
        action="store_false",
        help="Skip copying dataset CSV files into the Colab workspace.",
    )
    parser.add_argument(
        "--copy-data",
        dest="copy_data",
        action="store_true",
        help="Copy dataset CSV files from the repository into the Colab workspace (default).",
    )
    parser.set_defaults(copy_data=True)
    parser.add_argument(
        "--install-extras",
        choices=["none", "transformers", "ui", "all"],
        default="transformers",
        help="Install optional extras for training and/or the demo UI.",
    )
    parser.add_argument(
        "--install-pyright",
        action="store_true",
        help="Install Pyright inside the Colab runtime for static type checking.",
    )
    parser.add_argument(
        "--pip-verbose",
        action="store_true",
        help="Display pip output while installing dependencies (quiet mode is the default).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not running_in_colab():
        print("Warning: Colab environment not detected. Continuing anyway.")

    workspace_root = normalise_workspace(args.workspace)
    workspace_root, models_dir = stage_workspace(workspace_root, args.models_subdir)

    print(f"Workspace root: {workspace_root}")
    print(f"Models directory: {models_dir}")

    if args.copy_data:
        copied = copy_datasets(REPO_ROOT / DATA_DIR_NAME, workspace_root / DATA_DIR_NAME)
        if copied:
            print("Copied datasets:")
            for item in copied:
                print(f"  - {item}")
        else:
            print("No dataset files needed to be copied.")

    extras = extras_packages(args.install_extras)
    quiet = not args.pip_verbose
    if extras:
        install_packages(extras, quiet=quiet)

    if args.install_pyright:
        install_packages(["pyright"], quiet=quiet)

    print(
        (
            "Colab environment preparation complete. Update SOLARUS_COLAB_WORKSPACE to override the "
            "workspace location when invoking train_intent_classifier.py."
        )
    )


if __name__ == "__main__":
    main()
