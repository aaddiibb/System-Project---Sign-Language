"""
Kiseki — Core module for motion trajectory reconstruction.
Converts .npy motion features to 3D joint positions.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List


# ============================================================================
# Quaternion Operations (Pure NumPy - replaces torch dependency)
# ============================================================================

def quat_inv(q: np.ndarray) -> np.ndarray:
    """Invert quaternion(s). Shape: (*, 4)"""
    result = q.copy()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector(s) v by quaternion(s) q.
    q: (*, 4) quaternions [w, x, y, z]
    v: (*, 3) vectors
    Returns: (*, 3) rotated vectors
    """
    original_shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    
    qvec = q[:, 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


# ============================================================================
# Motion Reconstruction
# ============================================================================

def recover_root_transform(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover root rotation and position from motion features.
    
    Feature layout: [rot_vel(1), lin_vel(2:XZ), root_y(1), ...]
    
    Returns:
        r_rot_quat: (N, 4) root rotation quaternions
        r_pos: (N, 3) root positions
    """
    # Root rotation from angular velocity
    rot_vel = data[:, 0:1]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang / 2, axis=0)
    
    # Convert to quaternion [w, 0, y, 0]
    r_rot_quat = np.zeros((data.shape[0], 4))
    r_rot_quat[:, 0] = np.cos(r_rot_ang[:, 0])
    r_rot_quat[:, 2] = np.sin(r_rot_ang[:, 0])
    
    # Root position from linear velocity
    l_velocity = data[:, 1:3]
    r_pos = np.zeros((data.shape[0], 3))
    r_pos[1:, [0, 2]] = l_velocity[:-1]
    r_pos = quat_rotate(quat_inv(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = data[:, 3]  # Y from features
    
    return r_rot_quat, r_pos


def reconstruct_positions(features: np.ndarray, 
                         num_joints: int,
                         parents: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Reconstruct 3D joint positions from motion features.
    
    Args:
        features: (N, D) motion features
        num_joints: Number of skeleton joints
        parents: Parent indices for each joint (optional)
    
    Returns:
        positions: (N, J, 3) joint positions
        hierarchy: Dict mapping child -> parent indices
    """
    num_frames = features.shape[0]
    
    # Recover root
    r_rot_quat, r_pos = recover_root_transform(features)
    
    # Extract RIC positions
    ric_start = 4 + num_joints * 6
    ric_end = ric_start + num_joints * 3
    ric_data = features[:, ric_start:ric_end]
    ric_positions = ric_data.reshape(num_frames, num_joints, 3)
    
    # Convert to global
    r_rot_expanded = np.repeat(r_rot_quat[:, None, :], num_joints, axis=1)
    global_positions = quat_rotate(quat_inv(r_rot_expanded), ric_positions)
    global_positions[:, :, 0] += r_pos[:, 0:1]
    global_positions[:, :, 2] += r_pos[:, 2:3]
    
    # Build hierarchy
    if parents is None:
        parents = [-1] + list(range(num_joints - 1))
    
    hierarchy = {i: parents[i] if parents[i] >= 0 else None 
                 for i in range(num_joints)}
    
    return global_positions, hierarchy


# ============================================================================
# Minimal BVH Parser (replaces pymotion dependency)
# ============================================================================

def parse_bvh_hierarchy(bvh_path: str) -> Tuple[List[str], List[int]]:
    """
    Lightweight BVH parser - extracts joint names and parent indices.
    
    Skips End Site entries to match pymotion's behavior (End Sites are not
    real joints and should not be included in the joint list).
    
    Returns:
        joint_names: List of joint names
        parents: List of parent indices (-1 for root)
    """
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
    
    joint_names = []
    parents = []
    stack = []
    in_end_site = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('ROOT') or line.startswith('JOINT'):
            parts = line.split()
            joint_name = parts[1]
            joint_names.append(joint_name)
            
            parent_idx = stack[-1] if stack else -1
            parents.append(parent_idx)
            stack.append(len(joint_names) - 1)
            
        elif line.startswith('End Site'):
            # Skip End Sites - they are not real joints.
            # Flag so we ignore the matching closing brace.
            in_end_site = True
            
        elif line == '}':
            if in_end_site:
                # This closes the End Site block, not a real joint
                in_end_site = False
            elif stack:
                stack.pop()
                
        elif line.startswith('MOTION'):
            break
    
    return joint_names, parents


# ============================================================================
# Utilities
# ============================================================================

def find_bvh_file(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """Find a reference BVH file.
    
    Uses the bundled sample.bvh as default unless a different BVH is found
    in the provided search paths.
    """
    # If custom search paths are given, try those first
    if search_paths:
        for pattern in search_paths:
            for bvh_file in Path(".").glob(pattern):
                if bvh_file.exists():
                    return str(bvh_file)
    
    # Fall back to the bundled sample.bvh shipped with the package
    bundled_bvh = Path(__file__).parent / "sample.bvh"
    if bundled_bvh.exists():
        return str(bundled_bvh)
    
    # Last resort: search common dataset locations
    default_paths = [
        "dataset/*/*.bvh",
        "motion-s/*.bvh",
        "data/*/*.bvh",
    ]
    
    for pattern in default_paths:
        for bvh_file in Path(".").glob(pattern):
            if bvh_file.exists():
                return str(bvh_file)
    
    return None


def load_features(npy_path: str, 
                 norm_path: Optional[str] = None) -> np.ndarray:
    """Load and optionally denormalize motion features."""
    features = np.load(npy_path)
    
    if len(features.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {features.shape}")
    
    if norm_path:
        norm_data = np.load(norm_path)
        features = features * norm_data['std'] + norm_data['mean']
    
    return features


def get_bone_connections(hierarchy: Dict[int, Optional[int]]) -> List[Tuple[int, int]]:
    """Get list of (parent, child) bone pairs."""
    return [(parent, child) for child, parent in hierarchy.items() 
            if parent is not None]


# ============================================================================
# Joint Groups for Focused Views
# ============================================================================

JOINT_GROUPS = {
    'left_hand': list(range(20, 36)),
    'right_hand': list(range(39, 55)),
    'both_hands': list(range(20, 36)) + list(range(39, 55)),
    'left_arm': list(range(17, 36)),
    'right_arm': list(range(36, 55)),
    'both_arms': list(range(17, 55)),
    'upper_body': list(range(9, 55)),
    'fingers': list(range(21, 36)) + list(range(40, 55)),
}

FINGER_JOINTS = set(range(21, 36)) | set(range(40, 55))
HAND_JOINTS = set(range(20, 36)) | set(range(39, 55))


# ============================================================================
# Trail Presets (joint names to trace over time)
# ============================================================================

TRAIL_PRESETS = {
    'wrists': ['left_wrist', 'right_wrist'],
    'hands': ['left_wrist', 'right_wrist', 'left_index1', 'right_index1',
              'left_middle1', 'right_middle1'],
    'fingertips': ['left_index3', 'left_middle3', 'left_pinky3', 'left_ring3', 'left_thumb3',
                   'right_index3', 'right_middle3', 'right_pinky3', 'right_ring3', 'right_thumb3'],
    'feet': ['left_foot', 'right_foot'],
    'all_extremities': ['left_wrist', 'right_wrist', 'left_foot', 'right_foot', 'head'],
}


def resolve_joint_indices(joints, joint_names: List[str]) -> Optional[List[int]]:
    """
    Resolve joint names, indices, or presets to a list of joint indices.
    
    Args:
        joints: str (preset or single name), list of str/int, or None
        joint_names: List of joint names from BVH
    
    Returns:
        List of joint indices, or None
    """
    if joints is None:
        return None
    
    # If it's a preset string, expand it
    if isinstance(joints, str):
        if joints in TRAIL_PRESETS:
            joints = TRAIL_PRESETS[joints]
        else:
            joints = [joints]
    
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    
    indices = []
    for j in joints:
        if isinstance(j, int):
            if 0 <= j < len(joint_names):
                indices.append(j)
            else:
                print(f"Warning: joint index {j} out of range, skipping")
        elif isinstance(j, str):
            if j in name_to_idx:
                indices.append(name_to_idx[j])
            else:
                print(f"Warning: joint '{j}' not found in skeleton, skipping")
    
    return indices if indices else None