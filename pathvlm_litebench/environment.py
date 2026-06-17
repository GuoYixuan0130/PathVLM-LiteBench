from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from . import version

_REPORTED_PACKAGES = ("torch", "transformers", "numpy", "scikit-learn", "pillow")


def _safe_package_version(package_name: str) -> str | None:
    """Return an installed package version, or None if it is not installed."""
    try:
        return _package_version(package_name)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict:
    """
    Capture the runtime versions needed to reproduce a benchmark run.

    Records the toolkit version, the Python interpreter and platform, and the
    versions of the libraries that determine model behaviour (torch,
    transformers, numpy, scikit-learn, pillow). Embedding this in a run's
    metadata.json lets a result be tied to the exact stack that produced it.
    """
    packages = {
        package_name: _safe_package_version(package_name)
        for package_name in _REPORTED_PACKAGES
    }
    return {
        "pathvlm_litebench": version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
