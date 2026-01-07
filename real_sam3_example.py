"""
Real SAM 3 example using the actual model for text-prompted image segmentation.
"""

import os
import sys
import warnings

# Suppress CUDA warnings early
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment to prevent CUDA operations
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import SAM3 components
import sam3
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor
# Import visualization utils lazily to avoid potential import-time issues
# from sam3.visualization_utils import draw_box_on_image, plot_results

# Get the SAM3 root directory for assets
# Defer this until after imports to avoid issues
sam3_root = None
def get_sam3_root():
    global sam3_root
    if sam3_root is None:
        try:
            if hasattr(sam3, '__file__') and sam3.__file__:
                sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            else:
                sam3_root = os.path.join(os.path.dirname(__file__), "sam3")
        except (TypeError, AttributeError):
            sam3_root = os.path.join(os.path.dirname(__file__), "sam3")
    return sam3_root

# Setup device - force CPU to avoid MPS issues
# MPS (Metal) can cause device mismatch errors, so we'll use CPU for stability
device = "cpu"  # Force CPU for macOS compatibility
print(f"Using device: {device}")

# Enable TensorFloat-32 for Ampere GPUs if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_sam3_model():
    """Load the SAM3 model from Hugging Face."""
    print("Loading SAM3 model...")
    print("Note: This will download the model checkpoint from Hugging Face if not already cached.")
    print("You may need to authenticate with: hf auth login")
    
    # Suppress CUDA warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*CUDA.*")
    
    # Build model - this will automatically download from HF if needed
    # Try multiple paths for BPE file
    root = get_sam3_root()
    possible_bpe_paths = [
        os.path.join(root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(os.path.dirname(__file__), "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    ]
    
    bpe_path = None
    for path in possible_bpe_paths:
        if os.path.exists(path):
            bpe_path = path
            print(f"Using BPE path: {bpe_path}")
            break
    
    if bpe_path is None:
        # Last resort: let model builder use default
        print("Warning: Could not find BPE file, using model builder default")
        bpe_path = None
    
    print("Building model architecture...")
    try:
        # First, try to build without loading checkpoint to verify architecture works
        print("  Step 1: Building model architecture (without checkpoint)...")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",  # Force CPU on macOS
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,  # Don't download yet
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,  # Disable torch.compile which may cause issues on CPU
        )
        print("  ✓ Model architecture built successfully")
        
        # Now try to load checkpoint
        print("  Step 2: Loading model checkpoint from Hugging Face...")
        print("  (This requires access to https://huggingface.co/facebook/sam3)")
        try:
            checkpoint_path = download_ckpt_from_hf()
            print(f"  ✓ Checkpoint downloaded: {checkpoint_path}")
            
            # Load the checkpoint
            from sam3.model_builder import _load_checkpoint
            _load_checkpoint(model, checkpoint_path)
            print("  ✓ Checkpoint loaded")
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "GatedRepoError" in error_str or "not in the authorized list" in error_str:
                print(f"\n  ⚠️  Cannot download checkpoint: Access required")
                print(f"  Please visit https://huggingface.co/facebook/sam3 to request access")
                raise
            else:
                print(f"  ⚠️  Error loading checkpoint: {e}")
                print(f"  Continuing without checkpoint (model will not work properly)")
                raise
        
        print("Model loaded successfully!")
        return model
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        raise
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error during model building: {e}")
        import traceback
        traceback.print_exc()
        raise


def segment_with_text_prompt(image_path: str, text_prompt: str, model, processor):
    """
    Perform image segmentation using SAM 3 with a text prompt.
    
    Args:
        image_path: Path to the input image
        text_prompt: Text description of what to segment (e.g., "white dog", "red car")
        model: SAM3 model
        processor: SAM3 processor
    
    Returns:
        Segmentation results
    """
    print(f"\nProcessing image: {image_path}")
    print(f"Text prompt: '{text_prompt}'")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image size: {width}x{height}")
    
    # Set image in processor
    inference_state = processor.set_image(image)
    
    # Reset all prompts
    processor.reset_all_prompts(inference_state)
    
    # Set text prompt
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    # Extract results
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    if len(scores) > 0:
        print(f"Found {len(masks)} instances with scores: {scores.tolist()[:5]}{'...' if len(scores) > 5 else ''}")
        print(f"  Score range: {scores.min().item():.3f} - {scores.max().item():.3f}")
    else:
        print(f"Found {len(masks)} instances (no detections above threshold)")
    
    return {
        "masks": masks,
        "boxes": boxes,
        "scores": scores,
        "image": image,
        "prompt": text_prompt
    }


def visualize_results(result, output_path: str = None):
    """
    Visualize segmentation results.
    
    Args:
        result: Dictionary containing masks, boxes, scores, image, and prompt
        output_path: Path to save the visualization
    """
    image = result["image"]
    masks = result["masks"]
    boxes = result["boxes"]
    scores = result["scores"]
    prompt = result["prompt"]
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Segmented image with masks
    axes[1].imshow(img_array)
    
    # Draw masks
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        # Convert mask to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) == 3:
            mask = mask[0]  # Take first mask if multiple
        
        # Apply mask overlay
        mask_bool = mask > 0.5  # Threshold the mask
        overlay = img_array.copy()
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        # Draw bounding box
        if box is not None:
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            if len(box) == 4:
                x1, y1, x2, y2 = box
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               linewidth=2, edgecolor='yellow', facecolor='none')
                axes[1].add_patch(rect)
        
        # Apply overlay
        img_array = overlay
    
    axes[1].imshow(img_array)
    axes[1].set_title(f"Segmented: '{prompt}'\n{len(masks)} instances found", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function demonstrating SAM 3 usage with real model.
    """
    print("=" * 70)
    print("🎯 SAM 3 Real Model Example - Text-Prompted Image Segmentation")
    print("=" * 70)
    
    # Load model
    try:
        model = load_sam3_model()
        # Lower confidence threshold to detect more instances
        processor = Sam3Processor(model, device="cpu", confidence_threshold=0.3)
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ Error loading model: {error_str}")
        
        # Check for specific error types
        if "403" in error_str or "GatedRepoError" in error_str or "not in the authorized list" in error_str:
            print("\n" + "="*70)
            print("⚠️  ACCESS REQUIRED: You need to request access to the SAM3 model")
            print("="*70)
            print("\nSteps to get access:")
            print("1. Visit: https://huggingface.co/facebook/sam3")
            print("2. Click 'Agree and access repository'")
            print("3. Wait for approval (usually instant or a few minutes)")
            print("4. Make sure you're logged in: hf auth login")
            print("5. Run this script again")
            print("\n" + "="*70)
        elif "CUDA" in error_str or "cuda" in error_str.lower():
            print("\n⚠️  CUDA Error: This script is configured for CPU mode on macOS.")
            print("   If you see this error, there may be a compatibility issue.")
        else:
            print("\nTroubleshooting:")
            print("1. Make sure you have authenticated with Hugging Face:")
            print("   Run: hf auth login")
            print("2. Request access to the SAM3 model at:")
            print("   https://huggingface.co/facebook/sam3")
            print("3. Ensure you have sufficient disk space (model is ~3GB)")
            print("4. Check the full error above for more details")
        
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return
    
    # Check for demo image or use test image from SAM3 assets
    demo_image = "demo_image.jpg"
    test_image = os.path.join(get_sam3_root(), "assets", "images", "test_image.jpg")
    
    image_path = None
    if Path(demo_image).exists():
        image_path = demo_image
    elif Path(test_image).exists():
        image_path = test_image
    else:
        print(f"\n⚠️  No image found. Please provide an image path.")
        print(f"   Looking for: {demo_image} or {test_image}")
        return
    
    # Example text prompts - try variations for better detection
    # Note: More specific terms often work better (e.g., "child" vs "person")
    text_prompts = [
        "shoe",
        "person",  # Works with lower threshold (0.3)
        "child",   # Works better than "person" for this image
        "children",
    ]
    
    print(f"\n🖼️  Using image: {image_path}")
    print(f"📝 Testing {len(text_prompts)} text prompts...\n")
    
    # Process each prompt
    for i, prompt in enumerate(text_prompts, 1):
        print(f"\n[{i}/{len(text_prompts)}]")
        try:
            result = segment_with_text_prompt(image_path, prompt, model, processor)
            
            if len(result["masks"]) > 0:
                output_path = f"real_result_{i}_{prompt.replace(' ', '_')}.png"
                visualize_results(result, output_path)
            else:
                print(f"   ⚠️  No instances found for prompt: '{prompt}'")
        except Exception as e:
            print(f"   ❌ Error processing prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✨ Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

