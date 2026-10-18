"""
Kiseki (軌跡) — Lightweight motion trajectory visualization from .npy features.

A minimal-dependency package for converting motion feature files to 
animated skeleton videos.

Example:
    >>> from kiseki import visualize
    >>> visualize("motion.npy", focus_joints='both_hands', fixed_view='front')
    
    >>> from kiseki import compare
    >>> compare("generated.npy", "ground_truth.npy", mode="overlay")
"""

from .api import visualize
from .compare import compare
from .core import JOINT_GROUPS, TRAIL_PRESETS
from .visualize import VIEW_PRESETS

__version__ = "0.1.0"
__all__ = ['visualize', 'compare', 'JOINT_GROUPS', 'VIEW_PRESETS', 'TRAIL_PRESETS']
