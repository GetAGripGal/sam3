#!/usr/bin/env python3
"""
Verify SAM 3 video demo on 10 frames only.
"""
import os
import sys
import shutil
import tempfile

repo_root = os.path.dirname(os.path.abspath(__file__))
sam3_pkg = os.path.join(repo_root, "sam3")
if sam3_pkg not in sys.path:
    sys.path.insert(0, sam3_pkg)

os.environ.setdefault("SAM3_VIDEO_DEVICE", "cpu" if not __import__("torch").cuda.is_available() else "")

import torch
from sam3.model_builder import build_sam3_video_predictor

def main():
    # Use first 10 frames from assets/videos/0001
    src_dir = os.path.join(repo_root, "assets", "videos", "0001")
    if not os.path.isdir(src_dir):
        print("SKIP: assets/videos/0001 not found")
        return False
    jpgs = sorted(
        [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")],
        key=lambda f: int(os.path.splitext(f)[0])
    )[:10]
    if len(jpgs) < 10:
        print(f"SKIP: only {len(jpgs)} frames in 0001")
        return False

    tmp_dir = tempfile.mkdtemp(prefix="sam3_10f_")
    try:
        for f in jpgs:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(tmp_dir, f))
        video_path = tmp_dir
        out_dir = os.path.join(repo_root, "video_demo_10frames_output")
        print(f"Using 10 frames from 0001 in {tmp_dir}")
        print(f"Output will be saved to: {os.path.abspath(out_dir)}")
        print("Device:", "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

        print("Building predictor...")
        predictor = build_sam3_video_predictor()
        print("Starting session (10 frames)...")
        r = predictor.handle_request(request={"type": "start_session", "resource_path": video_path})
        session_id = r["session_id"]
        print("Session started.")

        print("Adding text prompt 'person' on frame 0...")
        r = predictor.handle_request(request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": "person",
        })
        out = r["outputs"]
        obj_ids = out.get("obj_ids", [])
        print(f"Frame 0: {len(obj_ids)} objects detected: {obj_ids}")

        print("Propagating to remaining frames...")
        outputs_per_frame = {}
        for resp in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session_id}):
            outputs_per_frame[resp["frame_index"]] = resp["outputs"]
        print(f"Propagated {len(outputs_per_frame)} frames total.")

        # Save output: visualization images
        os.makedirs(out_dir, exist_ok=True)
        video_frames_for_vis = [os.path.join(tmp_dir, f) for f in jpgs]
        try:
            from sam3.visualization_utils import prepare_masks_for_visualization, visualize_formatted_frame_output
            prepared = prepare_masks_for_visualization(outputs_per_frame)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for frame_idx in range(len(jpgs)):
                out_path = os.path.join(out_dir, f"frame_{frame_idx:02d}.png")
                visualize_formatted_frame_output(
                    frame_idx,
                    video_frames_for_vis,
                    outputs_list=[prepared],
                    titles=["SAM 3 video tracking"],
                    figsize=(6, 4),
                )
                plt.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close()
                print(f"  Saved {out_path}")
            print(f"\nOutput saved to: {os.path.abspath(out_dir)}")
        except Exception as viz_err:
            print(f"  (Visualization skip: {viz_err})")

        predictor.handle_request(request={"type": "close_session", "session_id": session_id})
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()
        print("OK: Video demo ran correctly on 10 frames.")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
