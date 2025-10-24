from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent.resolve()


def _ensure_pip_index(url: str) -> None:
    """Ensure the given index URL is available to pip during installation."""

    if not url:
        return
    cleaned = url.strip()
    if not cleaned:
        return
    primary = os.environ.get("PIP_INDEX_URL")
    if not primary:
        os.environ["PIP_INDEX_URL"] = cleaned
        return
    extra = os.environ.get("PIP_EXTRA_INDEX_URL", "")
    entries = [entry.strip() for entry in extra.split() if entry.strip()]
    if cleaned == primary or cleaned in entries:
        return
    entries.append(cleaned)
    os.environ["PIP_EXTRA_INDEX_URL"] = " ".join(entries)


def read_long_description() -> str:
    readme = ROOT / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "Solarus intent classification toolkit."


def base_requirements() -> list[str]:
    requires = [
        "numpy>=1.24",
        "torch>=2.1",
    ]
    variant_aliases = {
        "cpu": ("torch==2.3.1+cpu", "https://download.pytorch.org/whl/cpu"),
        "cu118": ("torch==2.3.1+cu118", "https://download.pytorch.org/whl/cu118"),
        "cu121": ("torch==2.3.1+cu121", "https://download.pytorch.org/whl/cu121"),
    }

    torch_override = os.environ.get("SOLARUS_TORCH_SPEC")
    torch_variant = os.environ.get("SOLARUS_TORCH_VARIANT", "").strip().lower()
    if not torch_override and torch_variant:
        if torch_variant in variant_aliases:
            torch_override, suggested_index = variant_aliases[torch_variant]
            os.environ.setdefault("SOLARUS_TORCH_INDEX_URL", suggested_index)
        else:
            raise SystemExit(
                f"Unknown SOLARUS_TORCH_VARIANT '{torch_variant}'. "
                f"Valid options: {', '.join(sorted(variant_aliases))}."
            )
    torch_index = os.environ.get("SOLARUS_TORCH_INDEX_URL")
    if torch_override:
        requires[-1] = torch_override
    if torch_index:
        _ensure_pip_index(torch_index)
    return requires


def collect_data_files() -> list[str]:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return []
    return [str(path.relative_to(ROOT)) for path in data_dir.glob("*.csv")]


def build_setup_kwargs() -> dict:
    extras = {
        "transformers": [
            "transformers>=4.34",
            "sentence-transformers>=2.2.2",
        ],
        "sklearn": ["scikit-learn>=1.3"],
        "ui": [
            "gradio>=4.0",
            "transformers>=4.34",
            "sentence-transformers>=2.2.2",
        ],
    }
    extras["all"] = sorted({req for group in extras.values() for req in group})

    shared_data = []
    data_files = collect_data_files()
    if data_files:
        shared_data.append(("share/solarus/data", data_files))

    metadata = {
        "name": "solarus-intent",
        "version": "0.1.0",
        "description": "Compact intent classification playground with advanced training utilities.",
        "long_description": read_long_description(),
        "long_description_content_type": "text/markdown",
        "author": "Solarus Project",
        "python_requires": ">=3.10",
        "license": "MIT",
        "packages": find_namespace_packages(include=["scripts", "scripts.*"]),
        "py_modules": ["train_intent_classifier"],
        "package_dir": {"": "."},
        "include_package_data": True,
        "data_files": shared_data,
        "install_requires": base_requirements(),
        "extras_require": extras,
        "entry_points": {
            "console_scripts": [
                "solarus-train=train_intent_classifier:main",
                "solarus-chat=scripts.orion_chat_ui:main",
                "solarus-update-dataset=scripts.update_intent_dataset:main",
            ]
        },
        "classifiers": [
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        "keywords": "intent-classification nlp pytorch self-training",
        "project_urls": {
            "Repository": "https://github.com/example/solarus",
            "Documentation": "https://github.com/example/solarus",
        },
    }
    return metadata


def main() -> None:
    if len(sys.argv) == 1:
        extra = os.environ.get("SOLARUS_AUTO_EXTRA", "all").strip()
        target = "."
        if extra and extra != "none":
            target = f".[{extra}]"
        print(f"Auto-installing Solarus package via pip install {target!s} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", target])
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Automatic installation failed with exit code {exc.returncode}. "
                "Re-run with an explicit command or set SOLARUS_AUTO_EXTRA=none to disable auto-install."
            ) from exc
        return
    setup(**build_setup_kwargs())


if __name__ == "__main__":
    main()
