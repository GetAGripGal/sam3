# SAM 3 Real Model Example

A working example of Segment Anything Model 3 (SAM 3) using the actual model (not a mock) for text-prompted image segmentation.

## ⚠️ Important: Access Required

**You must request access to the SAM3 model before running this script:**

1. Visit: https://huggingface.co/facebook/sam3
2. Click **"Agree and access repository"**
3. Wait for approval (usually instant or a few minutes)
4. Make sure you're authenticated: `hf auth login`

## Setup

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher (CPU version works on macOS)
- macOS or Linux (CUDA optional, but recommended for performance)

### Installation

1. **Clone and install SAM3** (if not already done):
   ```bash
   git clone https://github.com/facebookresearch/sam3
   cd sam3
   pip install -e .
   ```

2. **Install additional dependencies**:
   ```bash
   pip install opencv-python matplotlib pillow numpy
   ```

3. **Authenticate with Hugging Face**:
   ```bash
   hf auth login
   ```
   (You'll need a token from https://huggingface.co/settings/tokens)

4. **Request access to SAM3 model**:
   - Visit: https://huggingface.co/facebook/sam3
   - Click "Agree and access repository"
   - Wait for approval

## Usage

```bash
python real_sam3_example.py
```

The script will:
1. Download the SAM3 model checkpoint (~3GB) from Hugging Face (first run only)
2. Load the model on CPU (macOS compatible)
3. Process images with text prompts like "shoe", "person", "dog"
4. Generate segmentation visualizations

## Features

- ✅ **Real SAM3 Model**: Uses the actual Facebook SAM3 model, not a mock
- ✅ **CPU Compatible**: Works on macOS without CUDA
- ✅ **Automatic Download**: Downloads model checkpoint automatically
- ✅ **Text Prompts**: Segment objects using natural language
- ✅ **Visualization**: Generates side-by-side comparison images

## Troubleshooting

### "403 Forbidden" or "GatedRepoError"
- You need to request access at https://huggingface.co/facebook/sam3
- Make sure you're logged in: `hf auth login`

### "CUDA not available" warnings
- These are normal on macOS - the script uses CPU mode
- The warnings can be safely ignored

### Segmentation fault
- Make sure you have the latest PyTorch installed
- Try: `pip install --upgrade torch torchvision`

### Model download fails
- Check your internet connection
- Ensure you have ~3GB free disk space
- Verify Hugging Face authentication: `hf whoami`

## Example Output

The script processes images and generates files like:
- `real_result_1_shoe.png`
- `real_result_2_person.png`
- `real_result_3_dog.png`

Each file shows the original image and the segmented result with masks and bounding boxes.

## Model Information

- **Model**: SAM 3 (Segment Anything Model 3)
- **Size**: ~3GB checkpoint
- **Parameters**: 848M
- **Capabilities**: Text-prompted segmentation, visual prompts, video tracking

## Resources

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 Hugging Face](https://huggingface.co/facebook/sam3)
- [SAM3 Paper](https://arxiv.org/abs/2406.02963)
