#!/usr/bin/env python3
"""
Kiseki — Command-line interface for motion trajectory visualization.
"""

import argparse
import sys
from pathlib import Path

from .api import visualize
from .compare import compare


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kiseki",
        description="Kiseki (軌跡) — Visualize motion trajectories from .npy feature files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  kiseki -i motion.npy

  # Focus on hands with front view
  kiseki -i motion.npy --focus both_hands --view front

  # With trajectory trails on wrists
  kiseki -i motion.npy --trails wrists --trail-length 40

  # Render only frames 50-200
  kiseki -i motion.npy --start 50 --end 200

  # Compare two motions (overlay)
  kiseki -i generated.npy --compare ground_truth.npy --mode overlay

  # Compare side-by-side
  kiseki -i generated.npy --compare ground_truth.npy --mode side_by_side

  # With normalization and grid
  kiseki -i motion.npy --norm normalization.npz --grid
        """
    )
    
    # Input/output
    parser.add_argument("-i", "--input", required=True,
                       help="Input .npy motion file")
    parser.add_argument("-o", "--output", default=None,
                       help="Output video path (default: input_name.mp4)")
    parser.add_argument("--bvh", default=None,
                       help="Reference BVH file (auto-detected if omitted)")
    parser.add_argument("--norm", default=None,
                       help="Normalization .npz file")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--downsample", type=int, default=1,
                       help="Downsample factor (default: 1)")
    parser.add_argument("--title", default=None,
                       help="Video title")
    
    # Frame range
    parser.add_argument("--start", type=int, default=None,
                       help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame index (exclusive)")
    
    # View options
    parser.add_argument("--static", action="store_true",
                       help="Use static camera (no tracking)")
    parser.add_argument("--focus", default=None,
                       choices=['both_hands', 'left_hand', 'right_hand',
                               'both_arms', 'left_arm', 'right_arm',
                               'fingers', 'upper_body'],
                       help="Focus on specific body part")
    parser.add_argument("--view", default=None,
                       choices=['front', 'back', 'side', 'left_side',
                               'top', 'front_down', 'three_quarter'],
                       help="Fixed camera angle")
    parser.add_argument("--hand-size", type=float, default=8,
                       help="Point size for hand joints (default: 8)")
    
    # Trajectory trails
    parser.add_argument("--trails", default=None,
                       help="Joint names or preset for trajectory trails. "
                            "Presets: wrists, hands, fingertips, feet, all_extremities. "
                            "Or comma-separated joint names: left_wrist,right_wrist")
    parser.add_argument("--trail-length", type=int, default=30,
                       help="Number of past frames in trail (default: 30)")
    
    # Comparison
    parser.add_argument("--compare", default=None, metavar="NPY_FILE",
                       help="Second .npy file for motion comparison")
    parser.add_argument("--mode", default="overlay",
                       choices=['overlay', 'side_by_side'],
                       help="Comparison mode (default: overlay)")
    parser.add_argument("--label-a", default=None,
                       help="Label for first motion in comparison")
    parser.add_argument("--label-b", default=None,
                       help="Label for second motion in comparison")
    
    # Grid option
    parser.add_argument("--grid", action="store_true",
                       help="Also save frame grid as PNG")
    parser.add_argument("--grid-frames", type=int, default=9,
                       help="Number of frames in grid (default: 9)")
    
    args = parser.parse_args()
    
    try:
        # Parse trails argument
        trails = args.trails
        if trails and ',' in trails:
            trails = [t.strip() for t in trails.split(',')]
        
        # Comparison mode
        if args.compare:
            output_path = compare(
                npy_path_a=args.input,
                npy_path_b=args.compare,
                output_path=args.output,
                bvh_path=args.bvh,
                norm_path=args.norm,
                mode=args.mode,
                fps=args.fps,
                downsample=args.downsample,
                start_frame=args.start,
                end_frame=args.end,
                fixed_view=args.view,
                title=args.title,
                label_a=args.label_a,
                label_b=args.label_b,
            )
        else:
            # Standard visualization
            output_path = visualize(
                npy_path=args.input,
                output_path=args.output,
                bvh_path=args.bvh,
                norm_path=args.norm,
                fps=args.fps,
                downsample=args.downsample,
                tracking=not args.static,
                title=args.title,
                focus_joints=args.focus,
                fixed_view=args.view,
                hand_point_size=args.hand_size,
                save_grid=args.grid,
                grid_frames=args.grid_frames,
                start_frame=args.start,
                end_frame=args.end,
                trails=trails,
                trail_length=args.trail_length,
            )
        
        print("\n✓ Done!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
