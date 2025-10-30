# src/fargo2radmc3d/__init__.py
"""
fargo2radmc3d
=============

A converter and post-processing toolkit to transform FARGO3D or Dusty-FARGO
hydrodynamics simulations into RADMC-3D-ready model directories.

Typical usage:
    >>> from fargo2radmc3d import convert_fargo3d_snapshot
    >>> convert_fargo3d_snapshot("snapshot_dir", "radmc_model_dir")

Command-line interface:
    $ fargo2radmc3d convert --input snapshot_dir --out radmc_model_dir
"""

from __future__ import annotations

# Public API --------------------------------------------------------------
try:
    from .pipeline import convert_fargo3d_snapshot
except Exception:
    pass

# Optional re-exports (comment out if you want a tighter surface)
try:
    from .chemistry.pintelike import apply_pinte_switches
except Exception:
    pass

__all__ = [
    "convert_fargo3d_snapshot",
    "apply_pinte_switches",
]

# Package metadata --------------------------------------------------------
__version__ = "0.1.0"
__license__ = "MIT"

