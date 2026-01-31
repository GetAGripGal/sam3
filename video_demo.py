#!/usr/bin/env python3
"""
SAM 3 video demo using PR 264 (Apple Silicon support) style.
Runs text-prompted video segmentation and tracking on a short video.
"""
import os
import sys
import glob

# Run from repo root: ensure sam3 package is importable
repo_root = os.path.dirname(os.path.abspath(__file__))
sam3_pkg = os.path.join(repo_root, "sam3")
if sam3_pkg not in sys.path:
    sys.path.insert(0, sam3_pkg)

# PR 264: use CPU for video when no CUDA (avoids MPS segfaults on some Macs)
if not os.environ.get("SAM3_VIDEO_DEVICE"):
    os.environ["SAM3_VIDEO_DEVICE"] = "cpu" if not __import__("torch").cuda.is_available() else ""

import torch
import sam3
from sam3.model_builder import build_sam3_video_predictor
# Defer heavy viz imports to avoid potential segfault in sklearn/skimage on some systems
def _get_viz():
    from sam3.visualization_utils import (
        load_frame,
        prepare_masks_for_visualization,
        visualize_formatted_frame_output,
    )
    return load_frame, prepare_masks_for_visualization, visualize_formatted_frame_output

# PR 264: use single-device predictor (CPU/MPS when no CUDA)
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
if not os.path.isabs(sam3_root):
    sam3_root = os.path.abspath(os.path.join(repo_root, "sam3", sam3_root))

# Video: prefer JPEG frame folder (0001), else MP4
video_dir = os.path.join(repo_root, "assets", "videos", "0001")
video_mp4 = os.path.join(repo_root, "sam3", "assets", "videos", "bedroom.mp4")
if os.path.isdir(video_dir):
    video_path = video_dir
    print(f"Using video frames: {video_path}")
elif os.path.isfile(video_mp4):
    video_path = video_mp4
    print(f"Using video file: {video_path}")
else:
    print("No video found. Put assets/videos/0001/ (JPEG frames) or sam3/assets/videos/bedroom.mp4")
    sys.exit(1)

# Load frames for visualization (not used by model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        video_frames_for_vis.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        video_frames_for_vis.sort()
print(f"Frames for vis: {len(video_frames_for_vis)}")

def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def main():
    print("Device:", "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print("Building video predictor (PR 264 style, single device)...")
    predictor = build_sam3_video_predictor()
    print("Starting session on video...")
    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": video_path}
    )
    session_id = response["session_id"]
    print("Session ID:", session_id)

    # Text prompt on frame 0
    prompt_text = "person"
    frame_idx = 0
    print(f"Adding text prompt '{prompt_text}' on frame {frame_idx}...")
    response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "text": prompt_text,
        }
    )
    out = response["outputs"]
    print(f"Frame {frame_idx}: {len(out.get('obj_ids', []))} objects")

    # Propagate through video
    print("Propagating masks through video...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    print(f"Propagated to {len(outputs_per_frame)} frames")

    # Save a few visualizations (lazy import to avoid early segfault from viz deps)
    load_frame, prepare_masks_for_visualization, visualize_formatted_frame_output = _get_viz()
    os.makedirs(os.path.join(repo_root, "video_demo_output"), exist_ok=True)
    vis_frame_stride = max(1, len(outputs_per_frame) // 5)
    for i, frame_idx in enumerate(range(0, len(outputs_per_frame), vis_frame_stride)):
        if frame_idx >= len(video_frames_for_vis):
            break
        out_path = os.path.join(repo_root, "video_demo_output", f"frame_{frame_idx:04d}.png")
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=["SAM 3 video tracking"],
            figsize=(6, 4),
        )
        import matplotlib.pyplot as plt
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    predictor.handle_request(request={"type": "close_session", "session_id": session_id})
    if hasattr(predictor, "shutdown"):
        predictor.shutdown()
    print("Done.")

if __name__ == "__main__":
    main()
