"""
Kiseki — Visualization and animation functions for motion trajectory data.
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
    JOINT_GROUPS, 
    FINGER_JOINTS, 
    HAND_JOINTS
)


# ============================================================================
# Camera Presets
# ============================================================================

VIEW_PRESETS = {
    'front': (0, 90),        # Looking at character's front face
    'back': (0, -90),        # Looking at character's back
    'side': (0, 0),          # Right side view
    'left_side': (0, 180),   # Left side view
    'top': (90, -90),        # Top-down (looking down at front)
    'front_down': (15, 90),  # Slightly elevated front view
    'three_quarter': (20, 45),  # 3/4 view from right-front
}


# ============================================================================
# Notebook detection and display helpers
# ============================================================================

def _in_notebook() -> bool:
    """Return True if running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ('ZMQInteractiveShell',)
    except Exception:
        return False


def _display_animation(anim, fps: int):
    """Display a matplotlib animation inline in a Jupyter notebook."""
    from IPython.display import HTML, display as ipy_display
    
    try:
        # HTML5 video (requires ffmpeg, compact output)
        html = anim.to_html5_video()
    except Exception:
        # JS-based fallback (no ffmpeg needed, larger but always works)
        html = anim.to_jshtml()
    
    ipy_display(HTML(html))


def _save_animation(anim, output_path: Path, fps: int):
    """Save animation to a video file (mp4 or gif)."""
    if output_path.suffix.lower() == '.gif':
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
    else:
        try:
            writer = FFMpegWriter(
                fps=fps, 
                bitrate=1500,
                extra_args=['-vcodec', 'libx264', '-preset', 'ultrafast', 
                           '-pix_fmt', 'yuv420p']
            )
            anim.save(str(output_path), writer=writer, dpi=80)
        except Exception as e:
            print(f"FFMpeg failed: {e}. Saving as GIF.")
            output_path = output_path.with_suffix('.gif')
            writer = PillowWriter(fps=fps)
            anim.save(str(output_path), writer=writer)
    return output_path


# ============================================================================
# Animation Creation
# ============================================================================

def create_animation(positions: np.ndarray,
                    hierarchy: Dict[int, Optional[int]],
                    output_path: Optional[Union[str, Path]] = None,
                    fps: int = 30,
                    title: str = "Motion Visualization",
                    tracking: bool = True,
                    focus_joints: Optional[Union[List[int], str]] = None,
                    fixed_view: Optional[Union[Tuple[float, float], str]] = None,
                    hand_point_size: float = 8,
                    trail_indices: Optional[List[int]] = None,
                    trail_length: int = 30,
                    display: bool = False) -> Optional[Path]:
    """
    Create animated skeleton video.
    
    Args:
        positions: (N, J, 3) joint positions
        hierarchy: Joint parent mapping
        output_path: Output video path (ignored when display=True)
        fps: Frames per second
        title: Video title
        tracking: Camera follows root
        focus_joints: Joints to focus on (list or group name)
        fixed_view: Camera angle (tuple or preset name)
        hand_point_size: Point size for hand joints
        trail_indices: Joint indices to draw trajectory trails for
        trail_length: Number of past frames to show in trail
        display: If True, show inline in notebook instead of saving
    
    Returns:
        Path to saved video, or None when display=True
    """
    num_frames, num_joints = positions.shape[:2]
    bones = get_bone_connections(hierarchy)
    
    # Resolve focus joints
    if isinstance(focus_joints, str):
        focus_joints = JOINT_GROUPS.get(focus_joints)
    
    # Calculate view bounds
    if focus_joints:
        focus_pos = positions[:, focus_joints, :]
        min_vals = np.min(focus_pos, axis=(0, 1))
        max_vals = np.max(focus_pos, axis=(0, 1))
        center = (min_vals + max_vals) / 2
        radius = np.max(max_vals - min_vals) * 0.7
    else:
        root_pos = positions[0, 0]
        dists = np.linalg.norm(positions[0] - root_pos, axis=1)
        radius = np.max(dists) * 1.5
        center = root_pos
    
    # Resolve camera view
    if isinstance(fixed_view, str):
        fixed_view = VIEW_PRESETS.get(fixed_view, (15, 45))
    
    if focus_joints is not None:
        print(f"Focusing on {len(focus_joints)} joints, radius={radius:.1f}")
    if fixed_view is not None:
        print(f"Fixed view: elev={fixed_view[0]}, azim={fixed_view[1]}")
    
    # Setup figure
    fig = plt.figure(figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.grid(False)
    
    # Apply camera view — default to front view
    if fixed_view is not None:
        ax.view_init(elev=fixed_view[0], azim=fixed_view[1])
    else:
        ax.view_init(elev=VIEW_PRESETS['front'][0], azim=VIEW_PRESETS['front'][1])
    
    # Apply initial axis limits immediately
    if focus_joints is not None:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
    elif not tracking:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
    
    # Joint colors and sizes
    point_sizes = np.full(num_joints, 15.0)
    joint_colors = ['#c0392b'] * num_joints
    
    for j in range(num_joints):
        if j in FINGER_JOINTS:
            point_sizes[j] = hand_point_size
        elif j in HAND_JOINTS:
            point_sizes[j] = hand_point_size + 2
        
        if 20 <= j < 36:  # Left hand
            joint_colors[j] = '#3498db'
        elif 39 <= j < 55:  # Right hand
            joint_colors[j] = '#27ae60'
    
    # Initialize plot elements
    pos0 = positions[0]
    scatter = ax.scatter(pos0[:, 0], pos0[:, 2], pos0[:, 1],
                        c=joint_colors, s=point_sizes, depthshade=False)
    
    bone_collection = Line3DCollection([], colors='#2980b9', linewidths=1.5)
    ax.add_collection3d(bone_collection, autolim=False)
    
    frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                          color='#333', fontsize=12, family='monospace')
    
    # Pre-calculate bone indices
    bone_indices = np.array(bones)
    
    # Setup trajectory trails
    trail_collections = []
    TRAIL_COLORS = [
        [0.16, 0.50, 0.73],  # Blue
        [0.15, 0.68, 0.38],  # Green
        [0.91, 0.30, 0.24],  # Red
        [0.56, 0.27, 0.68],  # Purple
        [1.00, 0.60, 0.00],  # Orange
        [0.00, 0.74, 0.83],  # Cyan
        [0.80, 0.20, 0.60],  # Magenta
        [0.40, 0.65, 0.12],  # Olive
    ]
    if trail_indices is not None:
        print(f"Trails enabled for {len(trail_indices)} joints, length={trail_length}")
        for i, joint_idx in enumerate(trail_indices):
            base_color = TRAIL_COLORS[i % len(TRAIL_COLORS)]
            tc = Line3DCollection([], linewidths=2.5)
            ax.add_collection3d(tc, autolim=False)
            trail_collections.append((joint_idx, base_color, tc))
    
    def _apply_view():
        """Apply camera view angle."""
        if fixed_view is not None:
            ax.view_init(elev=fixed_view[0], azim=fixed_view[1])
        else:
            ax.view_init(elev=VIEW_PRESETS['front'][0], azim=VIEW_PRESETS['front'][1])
    
    def _apply_limits():
        """Apply axis limits."""
        if focus_joints is not None:
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(center[1] - radius, center[1] + radius)
        elif not tracking:
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(center[1] - radius, center[1] + radius)
    
    def init():
        _apply_view()
        _apply_limits()
        return scatter, bone_collection, frame_text
    
    def update(frame):
        pos = positions[frame]
        
        # Update joints
        scatter._offsets3d = (pos[:, 0], pos[:, 2], pos[:, 1])
        
        # Update bones
        p_idx, c_idx = bone_indices[:, 0], bone_indices[:, 1]
        segments = np.stack([pos[p_idx][:, [0, 2, 1]], 
                            pos[c_idx][:, [0, 2, 1]]], axis=1)
        bone_collection.set_segments(segments)
        
        # Update trajectory trails
        for joint_idx, base_color, tc in trail_collections:
            start = max(0, frame - trail_length)
            trail_pos = positions[start:frame + 1, joint_idx]  # (T, 3)
            
            if len(trail_pos) > 1:
                # Swap Y/Z for matplotlib coordinate system
                points = trail_pos[:, [0, 2, 1]]
                segs = np.stack([points[:-1], points[1:]], axis=1)
                
                # RGBA colors with fading alpha (old=faint, new=bright)
                n_segs = len(segs)
                colors = np.zeros((n_segs, 4))
                colors[:, :3] = base_color
                colors[:, 3] = np.linspace(0.05, 0.9, n_segs)
                
                tc.set_segments(segs)
                tc.set_color(colors)
            else:
                tc.set_segments([])
        
        # Update camera limits per frame
        if focus_joints is not None:
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(center[1] - radius, center[1] + radius)
        elif tracking:
            target = pos[0]
            ax.set_xlim(target[0] - radius, target[0] + radius)
            ax.set_ylim(target[2] - radius, target[2] + radius)
            ax.set_zlim(target[1] - radius, target[1] + radius)
        
        # Re-apply view angle every frame
        if fixed_view is not None:
            ax.view_init(elev=fixed_view[0], azim=fixed_view[1])
        
        frame_text.set_text(f'Frame: {frame}/{num_frames-1}')
        return scatter, bone_collection, frame_text
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=num_frames, interval=1000/fps, blit=False)
    ax.set_title(title, pad=20)
    
    # Display inline or save to file
    if display:
        _display_animation(anim, fps)
        plt.close(fig)
        return None
    
    output_path = Path(output_path)
    result = _save_animation(anim, output_path, fps)
    plt.close(fig)
    return result


def create_frame_grid(positions: np.ndarray,
                     hierarchy: Dict[int, Optional[int]],
                     output_path: Union[str, Path],
                     num_frames: int = 9,
                     title: str = "Motion Frames") -> Path:
    """Create a grid of frames from motion sequence."""
    total_frames = positions.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    bones = get_bone_connections(hierarchy)
    
    # Grid layout
    cols = min(3, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    # View bounds
    min_vals = np.min(positions, axis=(0, 1))
    max_vals = np.max(positions, axis=(0, 1))
    center = (min_vals + max_vals) / 2
    limit = np.max(max_vals - min_vals) * 0.6
    
    # Create figure
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        pos = positions[frame_idx]
        
        # Draw
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c='#e74c3c', s=6)
        for parent, child in bones:
            p, c = pos[parent], pos[child]
            ax.plot([p[0], c[0]], [p[2], c[2]], [p[1], c[1]],
                   c='#2c3e50', lw=1.2)
        
        ax.set_xlim(center[0] - limit, center[0] + limit)
        ax.set_ylim(center[2] - limit, center[2] + limit)
        ax.set_zlim(center[1] - limit, center[1] + limit)
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=10, azim=90)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return Path(output_path)
