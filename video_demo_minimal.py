#!/usr/bin/env python3
"""Minimal video demo to find where crash occurs."""
import os
import sys
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, "sam3"))

# Force CPU to avoid MPS segfault
os.environ["SAM3_VIDEO_DEVICE"] = "cpu"

import torch
print("1. torch ok, cuda:", torch.cuda.is_available(), "mps:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

from sam3.model_builder import build_sam3_video_predictor
print("2. import build_sam3_video_predictor ok")

print("3. Building predictor...")
predictor = build_sam3_video_predictor()
print("4. Predictor built, device:", getattr(predictor, "device", "?"))

video_path = os.path.join(repo_root, "assets", "videos", "0001")
if not os.path.isdir(video_path):
    print("No video dir at", video_path)
    sys.exit(0)
print("5. Starting session...")
r = predictor.handle_request(request={"type": "start_session", "resource_path": video_path})
print("6. Session started:", r.get("session_id"))
