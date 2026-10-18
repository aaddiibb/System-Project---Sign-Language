"""
Kiseki — Motion comparison visualization (overlay and side-by-side).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List

from .core import (
    get_bone_connections,
    load_features,
    find_bvh_file,
    parse_bvh_hierarchy,
    reconstruct_positions,
)
from .visualize import VIEW_PRESETS, _display_animation, _save_animation


# Color schemes for the two motions
SCHEME_A = {'joints': '#2980b9', 'bones': '#3498db'}  # Blue
SCHEME_B = {'joints': '#e74c3c', 'bones': '#e67e22'}  # Red/Orange


def _compute_shared_bounds(positions_a: np.ndarray,
                           positions_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute view bounds that fit both motions."""
    all_pos = np.concatenate([positions_a, positions_b], axis=0)
    min_vals = np.min(all_pos, axis=(0, 1))
    max_vals = np.max(all_pos, axis=(0, 1))
    center = (min_vals + max_vals) / 2
    radius = np.max(max_vals - min_vals) * 0.7
    return center, radius


def _setup_ax(ax, center, radius, view):
    """Configure a 3D axis with shared bounds and view."""
    ax.set_facecolor('white')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.view_init(elev=view[0], azim=view[1])


def _make_skeleton(ax, num_joints, scheme, alpha=0.85):
    """Create scatter + bone collection for one skeleton."""
    scatter = ax.scatter([], [], [],
                         c=scheme['joints'], s=10,
                         alpha=alpha, depthshade=False)
    bones = Line3DCollection([], colors=scheme['bones'],
                             linewidths=1.5, alpha=alpha * 0.8)
    ax.add_collection3d(bones, autolim=False)
    return scatter, bones


def _update_skeleton(scatter, bone_coll, bone_indices, pos):
    """Update one skeleton for a single frame."""
    scatter._offsets3d = (pos[:, 0], pos[:, 2], pos[:, 1])
    p_idx, c_idx = bone_indices[:, 0], bone_indices[:, 1]
    segs = np.stack([pos[p_idx][:, [0, 2, 1]],
                     pos[c_idx][:, [0, 2, 1]]], axis=1)
    bone_coll.set_segments(segs)


def create_comparison_animation(
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        hierarchy: Dict[int, Optional[int]],
        output_path: Optional[Union[str, Path]] = None,
        mode: str = "overlay",
        fps: int = 30,
        title: Optional[str] = None,
        fixed_view: Optional[Union[Tuple[float, float], str]] = None,
        label_a: str = "Motion A",
        label_b: str = "Motion B",
        display: bool = False) -> Optional[Path]:
    """
    Create comparison animation of two motions.
    
    Args:
        positions_a: (N, J, 3) first motion positions
        positions_b: (M, J, 3) second motion positions
        hierarchy: Joint parent mapping
        output_path: Output video path (ignored when display=True)
        mode: "overlay" (both on same axes) or "side_by_side" (two panels)
        fps: Frames per second
        title: Video title
        fixed_view: Camera view preset string or (elev, azim) tuple
        label_a: Label for first motion
        label_b: Label for second motion
        display: If True, show inline in notebook instead of saving
    
    Returns:
        Path to saved video, or None when display=True
    """
    # Trim to same length
    num_frames = min(positions_a.shape[0], positions_b.shape[0])
    positions_a = positions_a[:num_frames]
    positions_b = positions_b[:num_frames]
    num_joints = positions_a.shape[1]
    
    bones = get_bone_connections(hierarchy)
    bone_indices = np.array(bones)
    
    # Resolve view
    if isinstance(fixed_view, str):
        view = VIEW_PRESETS.get(fixed_view, VIEW_PRESETS['front'])
    elif fixed_view is not None:
        view = fixed_view
    else:
        view = VIEW_PRESETS['front']
    
    center, radius = _compute_shared_bounds(positions_a, positions_b)
    
    print(f"Comparison mode: {mode}")
    print(f"Frames: {num_frames} (trimmed to shorter motion)")
    
    if mode == "side_by_side":
        fig = plt.figure(figsize=(20, 10), dpi=100)
        fig.patch.set_facecolor('white')
        ax_a = fig.add_subplot(121, projection='3d')
        ax_b = fig.add_subplot(122, projection='3d')
        
        _setup_ax(ax_a, center, radius, view)
        _setup_ax(ax_b, center, radius, view)
        ax_a.set_title(label_a, fontsize=14, color=SCHEME_A['joints'],
                       fontweight='bold')
        ax_b.set_title(label_b, fontsize=14, color=SCHEME_B['joints'],
                       fontweight='bold')
        
        scat_a, bones_a = _make_skeleton(ax_a, num_joints, SCHEME_A, alpha=1.0)
        scat_b, bones_b = _make_skeleton(ax_b, num_joints, SCHEME_B, alpha=1.0)
        
        frame_text = fig.text(0.5, 0.02, '', ha='center',
                              fontsize=12, family='monospace')
        
        def update(frame):
            _update_skeleton(scat_a, bones_a, bone_indices, positions_a[frame])
            _update_skeleton(scat_b, bones_b, bone_indices, positions_b[frame])
            _setup_ax(ax_a, center, radius, view)
            _setup_ax(ax_b, center, radius, view)
            frame_text.set_text(f'Frame: {frame}/{num_frames - 1}')
            return scat_a, bones_a, scat_b, bones_b, frame_text
    
    else:  # overlay
        fig = plt.figure(figsize=(10, 10), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        _setup_ax(ax, center, radius, view)
        
        scat_a, bones_a = _make_skeleton(ax, num_joints, SCHEME_A, alpha=0.85)
        scat_b, bones_b = _make_skeleton(ax, num_joints, SCHEME_B, alpha=0.85)
        
        # Legend
        ax.plot([], [], c=SCHEME_A['joints'], label=label_a, lw=3)
        ax.plot([], [], c=SCHEME_B['joints'], label=label_b, lw=3)
        ax.legend(loc='upper right', fontsize=11)
        
        frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                               color='#333', fontsize=12, family='monospace')
        
        def update(frame):
            _update_skeleton(scat_a, bones_a, bone_indices, positions_a[frame])
            _update_skeleton(scat_b, bones_b, bone_indices, positions_b[frame])
            _setup_ax(ax, center, radius, view)
            frame_text.set_text(f'Frame: {frame}/{num_frames - 1}')
            return scat_a, bones_a, scat_b, bones_b, frame_text
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=1000 / fps, blit=False)
    
    # Display inline or save to file
    if display:
        _display_animation(anim, fps)
        plt.close(fig)
        return None
    
    output_path = Path(output_path)
    result = _save_animation(anim, output_path, fps)
    plt.close(fig)
    return result


def compare(npy_path_a: Union[str, Path],
            npy_path_b: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
            bvh_path: Optional[Union[str, Path]] = None,
            norm_path: Optional[Union[str, Path]] = None,
            mode: str = "overlay",
            fps: int = 30,
            downsample: int = 1,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fixed_view: Optional[Union[Tuple[float, float], str]] = None,
            title: Optional[str] = None,
            label_a: Optional[str] = None,
            label_b: Optional[str] = None,
            display: bool = False) -> Optional[Path]:
    """
    Compare two .npy motion files as an animated video or display inline.
    
    Args:
        npy_path_a: First motion .npy file
        npy_path_b: Second motion .npy file
        output_path: Output video path (default: compare_A_vs_B.mp4 in cwd).
                     Ignored when display=True.
        bvh_path: Reference BVH file (auto-detected if None)
        norm_path: Normalization file for denormalization
        mode: "overlay" (both on same axes) or "side_by_side" (two panels)
        fps: Frames per second
        downsample: Downsample factor
        start_frame: Start frame (inclusive)
        end_frame: End frame (exclusive)
        fixed_view: Camera view preset or (elev, azim) tuple
        title: Video title
        label_a: Label for first motion
        label_b: Label for second motion
        display: If True, display inline in notebook instead of saving
    
    Returns:
        Path to saved video, or None when display=True
    
    Example:
        >>> from kiseki import compare
        >>> compare("generated.npy", "ground_truth.npy", mode="overlay")
        >>> compare("a.npy", "b.npy", mode="side_by_side", fixed_view='front')
        >>> # Display inline in a notebook:
        >>> compare("a.npy", "b.npy", display=True)
    """
    npy_path_a = Path(npy_path_a)
    npy_path_b = Path(npy_path_b)
    
    if not display:
        if output_path is None:
            output_path = Path.cwd() / f"compare_{npy_path_a.stem}_vs_{npy_path_b.stem}.mp4"
        else:
            output_path = Path(output_path)
    
    if label_a is None:
        label_a = npy_path_a.stem
    if label_b is None:
        label_b = npy_path_b.stem
    if title is None:
        title = f"{label_a}  vs  {label_b}"
    
    # Find BVH
    if bvh_path is None:
        bvh_path = find_bvh_file()
        if not bvh_path:
            raise FileNotFoundError(
                "No BVH file found. Please provide bvh_path argument."
            )
    
    print(f"Using BVH: {bvh_path}")
    
    # Load & reconstruct both motions
    joint_names, parents = parse_bvh_hierarchy(str(bvh_path))
    num_joints = len(joint_names)
    
    features_a = load_features(str(npy_path_a), norm_path)
    features_b = load_features(str(npy_path_b), norm_path)
    print(f"Motion A: {features_a.shape[0]} frames | "
          f"Motion B: {features_b.shape[0]} frames")
    
    positions_a, hierarchy = reconstruct_positions(features_a, num_joints, parents)
    positions_b, _ = reconstruct_positions(features_b, num_joints, parents)
    
    # Frame range
    if start_frame is not None or end_frame is not None:
        s = start_frame or 0
        e_a = min(end_frame, positions_a.shape[0]) if end_frame else positions_a.shape[0]
        e_b = min(end_frame, positions_b.shape[0]) if end_frame else positions_b.shape[0]
        positions_a = positions_a[s:e_a]
        positions_b = positions_b[s:e_b]
        print(f"Frame range: {s} -> A:{e_a}, B:{e_b}")
    
    # Downsample
    if downsample > 1:
        positions_a = positions_a[::downsample]
        positions_b = positions_b[::downsample]
    
    # Create comparison animation
    print("Creating comparison animation...")
    result_path = create_comparison_animation(
        positions_a, positions_b, hierarchy, output_path,
        mode=mode, fps=fps, title=title, fixed_view=fixed_view,
        label_a=label_a, label_b=label_b, display=display,
    )
    
    if result_path is not None:
        print(f"Saved: {result_path}")
    
    return result_path
