"""
Kiseki — Main API for motion trajectory visualization.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np

from .core import (
    load_features,
    find_bvh_file,
    parse_bvh_hierarchy,
    reconstruct_positions,
    resolve_joint_indices,
)
from .visualize import create_animation, create_frame_grid


def visualize(npy_path: Union[str, Path],
             output_path: Optional[Union[str, Path]] = None,
             bvh_path: Optional[Union[str, Path]] = None,
             norm_path: Optional[Union[str, Path]] = None,
             fps: int = 30,
             downsample: int = 1,
             tracking: bool = True,
             title: Optional[str] = None,
             focus_joints: Optional[Union[List[int], str]] = None,
             fixed_view: Optional[Union[Tuple[float, float], str]] = None,
             hand_point_size: float = 8,
             save_grid: bool = False,
             grid_frames: int = 9,
             start_frame: Optional[int] = None,
             end_frame: Optional[int] = None,
             trails: Optional[Union[List[str], str]] = None,
             trail_length: int = 30,
             display: bool = False) -> Optional[Path]:
    """
    Convert .npy motion file to animated video or display inline in a notebook.
    
    Args:
        npy_path: Input .npy motion file
        output_path: Output video path. Defaults to <stem>.mp4 in the current
                     working directory. Ignored when display=True.
        bvh_path: Reference BVH file (auto-detected if None)
        norm_path: Normalization file for denormalization
        fps: Frames per second
        downsample: Downsample factor
        tracking: Camera follows root
        title: Video title
        focus_joints: Focus on specific joints (list or group name)
                     Options: 'both_hands', 'left_hand', 'right_hand',
                             'both_arms', 'fingers', 'upper_body'
        fixed_view: Fixed camera view (tuple or preset)
                   Presets: 'front', 'back', 'side', 'left_side', 'top',
                           'front_down', 'three_quarter'
        hand_point_size: Point size for hand joints
        save_grid: Also save frame grid image
        grid_frames: Number of frames in grid
        start_frame: Start frame index (inclusive, applied after reconstruction)
        end_frame: End frame index (exclusive, applied after reconstruction)
        trails: Joint names or preset to draw trajectory trails for
                Presets: 'wrists', 'hands', 'fingertips', 'feet', 'all_extremities'
                Or list of joint names: ['left_wrist', 'right_wrist']
        trail_length: Number of past frames visible in each trail (default: 30)
        display: If True, display the animation inline in a Jupyter notebook
                 instead of saving to a file.
    
    Returns:
        Path to saved video, or None when display=True.
        
    Example:
        >>> from kiseki import visualize
        >>> visualize("motion.npy", focus_joints='both_hands', fixed_view='front')
        >>> visualize("motion.npy", trails='wrists', trail_length=30)
        >>> visualize("motion.npy", start_frame=50, end_frame=200)
        >>> # Display inline in a notebook (no file saved):
        >>> visualize("motion.npy", display=True)
    """
    # Setup paths
    npy_path = Path(npy_path)
    
    if not display:
        if output_path is None:
            # Write to current working directory, not the input file's directory
            output_path = Path.cwd() / npy_path.with_suffix('.mp4').name
        else:
            output_path = Path(output_path)
    
    if title is None:
        title = f"Motion: {npy_path.stem}"
    
    # Find BVH
    if bvh_path is None:
        bvh_path = find_bvh_file()
        if not bvh_path:
            raise FileNotFoundError(
                "No BVH file found. Please provide bvh_path argument."
            )
    
    print(f"Using BVH: {bvh_path}")
    
    # Load data
    features = load_features(str(npy_path), norm_path)
    print(f"Loaded: {features.shape[0]} frames, {features.shape[1]} features")
    
    # Parse BVH hierarchy
    joint_names, parents = parse_bvh_hierarchy(str(bvh_path))
    num_joints = len(joint_names)
    
    # Reconstruct
    print("Reconstructing skeleton...")
    positions, hierarchy = reconstruct_positions(features, num_joints, parents)
    print(f"Reconstructed: {positions.shape[0]} frames, {positions.shape[1]} joints")
    
    # Frame range selection
    if start_frame is not None or end_frame is not None:
        s = start_frame or 0
        e = end_frame or positions.shape[0]
        e = min(e, positions.shape[0])
        positions = positions[s:e]
        print(f"Frame range: {s}→{e} ({positions.shape[0]} frames)")
    
    # Downsample
    if downsample > 1:
        positions = positions[::downsample]
        print(f"Downsampled to {positions.shape[0]} frames")
    
    # Resolve trail joint names → indices
    trail_indices = resolve_joint_indices(trails, joint_names)
    
    # Create animation
    print("Creating animation...")
    result_path = create_animation(
        positions, hierarchy, output_path,
        fps=fps, title=title, tracking=tracking,
        focus_joints=focus_joints, fixed_view=fixed_view,
        hand_point_size=hand_point_size,
        trail_indices=trail_indices, trail_length=trail_length,
        display=display,
    )
    
    if result_path is not None:
        print(f"Saved: {result_path}")
    
    # Optional grid
    if save_grid and not display:
        grid_path = output_path.with_name(f"{output_path.stem}_frames.png")
        create_frame_grid(positions, hierarchy, grid_path, 
                         num_frames=grid_frames, title=title)
        print(f"Saved grid: {grid_path}")
    
    return result_path


__all__ = ['visualize']
