from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from setuptools import find_namespace_packages, setup  # type: ignore[import-not-found]

ROOT = Path(__file__).parent.resolve()


class SetupArguments(TypedDict, total=False):
    """Typed representation of the keyword arguments accepted by :func:`setup`."""

    name: str
    version: str
    description: str
    long_description: str
    long_description_content_type: str
    author: str
    python_requires: str
    license: str
    packages: list[str]
    py_modules: list[str]
    package_dir: dict[str, str]
    include_package_data: bool
    data_files: list[tuple[str, list[str]]]
    install_requires: list[str]
    extras_require: dict[str, list[str]]
    entry_points: dict[str, list[str]]
    classifiers: list[str]
    keywords: str
    project_urls: dict[str, str]


def _append_extra_index(url: str) -> None:
    if not url:
        return
    cleaned = url.strip()
    if not cleaned:
        return
    extra_entries: list[str] = [
        entry.strip()
        for entry in os.environ.get("PIP_EXTRA_INDEX_URL", "").split()
        if entry.strip()
    ]
    if cleaned in extra_entries:
        return
    extra_entries.append(cleaned)
    os.environ["PIP_EXTRA_INDEX_URL"] = " ".join(extra_entries)


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
        _append_extra_index("https://pypi.org/simple")
        return
    if cleaned == primary:
        return
    _append_extra_index(cleaned)


def read_long_description() -> str:
    readme = ROOT / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "Solarus intent classification toolkit."


def base_requirements() -> list[str]:
    requires: list[str] = [
        "numpy>=1.24",
        "torch>=2.1",
    ]
    variant_aliases: dict[str, dict[str, str]] = {
        "cpu": {"spec": "torch>=2.1", "index": "https://download.pytorch.org/whl/cpu"},
        "cu118": {"spec": "torch>=2.1", "index": "https://download.pytorch.org/whl/cu118"},
        "cu121": {"spec": "torch>=2.1", "index": "https://download.pytorch.org/whl/cu121"},
    }

    torch_override = os.environ.get("SOLARUS_TORCH_SPEC")
    torch_variant = os.environ.get("SOLARUS_TORCH_VARIANT", "").strip().lower()
    torch_index = os.environ.get("SOLARUS_TORCH_INDEX_URL")
    if not torch_override and torch_variant:
        alias = variant_aliases.get(torch_variant)
        if alias is None:
            raise SystemExit(
                f"Unknown SOLARUS_TORCH_VARIANT '{torch_variant}'. "
                f"Valid options: {', '.join(sorted(variant_aliases))}."
            )
        torch_override = alias["spec"]
        if not torch_index:
            torch_index = alias["index"]
    if not torch_index and torch_variant:
        alias = variant_aliases.get(torch_variant)
        if alias is not None:
            torch_index = alias["index"]
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


def build_setup_kwargs() -> SetupArguments:
    extras: dict[str, list[str]] = {
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

    shared_data: list[tuple[str, list[str]]] = []
    data_files = collect_data_files()
    if data_files:
        shared_data.append(("share/solarus/data", data_files))

    metadata: SetupArguments = {
        "name": "solarus-intent",
        "version": "0.1.0",
        "description": "Compact intent classification playground with advanced training utilities.",
        "long_description": read_long_description(),
        "long_description_content_type": "text/markdown",
        "author": "Solarus Project",
        "python_requires": ">=3.10",
        "license": "MIT",
        "packages": list(find_namespace_packages(include=["scripts", "scripts.*"])),
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
    kwargs = build_setup_kwargs()
    if len(sys.argv) == 1:
        auto_extra = os.environ.get("SOLARUS_AUTO_EXTRA", "all").strip().lower()
        if auto_extra and auto_extra != "none":
            target = f".[{auto_extra}]" if auto_extra != "all" else ".[all]"
            print(f"Attempting automatic dependency install via 'pip install {target}' ...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", target],
                check=False,
            )
            if result.returncode != 0:
                print(
                    "Warning: automatic 'pip install' failed "
                    f"(exit code {result.returncode}). Continue by installing dependencies manually "
                    "or re-running with SOLARUS_AUTO_EXTRA=none to skip this step."
                )
        # Provide a non-invasive default setuptools command when none was supplied.
        sys.argv.append("egg_info")
    setup(**kwargs)


if __name__ == "__main__":
    main()
