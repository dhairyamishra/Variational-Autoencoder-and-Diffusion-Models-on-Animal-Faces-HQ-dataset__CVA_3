# Diffusion Model - Local GPU Training Guide

This guide will help you set up and train the diffusion model on your local machine with GPU acceleration.

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for dataset + models

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: Compatible with your GPU (CUDA 11.8+ recommended)
- **Windows**: This guide is optimized for Windows

## 🚀 Quick Start

### Step 1: Install Dependencies

First, make sure you have PyTorch with CUDA support installed:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install other requirements:

```bash
pip install -r requirements.txt
```

### Step 2: Run Setup Script

Run the setup script to verify your GPU and download the dataset:

```bash
python setup_local.py
```

This script will:
- ✅ Check if CUDA is available
- ✅ Test your GPU
- ✅ Verify all dependencies
- ✅ Create necessary directories
- ✅ Help you download the AFHQ dataset (6.5GB)

### Step 3: Start Training

#### Option A: Use the Batch Script (Easiest)

Simply double-click `train_local.bat` or run:

```bash
train_local.bat
```

#### Option B: Use Python Directly

```bash
# Quick test (small model, fast)
python main.py --time_steps 50 --train_steps 1000 --batch_size 16 --image_size 32 --unet_dim 16 --data_class cat

# Standard training (recommended)
python main.py --time_steps 500 --train_steps 50000 --batch_size 32 --image_size 128 --unet_dim 64 --data_class cat

# High quality (requires more GPU memory)
python main.py --time_steps 1000 --train_steps 100000 --batch_size 16 --image_size 256 --unet_dim 128 --data_class cat
```

## 📊 Training Parameters

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--time_steps` | Number of diffusion steps | 500 | 50-1000 |
| `--train_steps` | Total training iterations | 50000 | 10000-100000 |
| `--batch_size` | Batch size | 32 | 8-64 (depends on GPU) |
| `--image_size` | Image resolution | 128 | 32-256 |
| `--unet_dim` | UNet base dimension | 64 | 16-128 |
| `--learning_rate` | Learning rate | 2e-5 | 1e-5 to 1e-4 |
| `--data_class` | Animal class | cat | cat, dog, wild, all |
| `--save_folder` | Output directory | ./results_afhq | any path |
| `--save_and_sample_every` | Save interval | 1000 | 500-5000 |

### GPU Memory Considerations

**If you get CUDA Out of Memory errors**, try:

1. **Reduce batch size**: `--batch_size 16` or `--batch_size 8`
2. **Reduce image size**: `--image_size 64` or `--image_size 32`
3. **Reduce model size**: `--unet_dim 32` or `--unet_dim 16`
4. **Reduce UNet depth**: `--unet_dim_mults 1 2 4` (instead of `1 2 4 8`)

Example for 4GB GPU:
```bash
python main.py --batch_size 8 --image_size 64 --unet_dim 32 --unet_dim_mults 1 2 4
```

Example for 8GB GPU:
```bash
python main.py --batch_size 16 --image_size 128 --unet_dim 64 --unet_dim_mults 1 2 4 8
```

Example for 16GB+ GPU:
```bash
python main.py --batch_size 32 --image_size 256 --unet_dim 128 --unet_dim_mults 1 2 4 8
```

## 📁 Project Structure

```
A3_PartB/
├── main.py                 # Main training script
├── trainer.py              # Training loop implementation
├── diffusion.py            # Diffusion model implementation
├── unet.py                 # UNet architecture
├── setup_local.py          # Setup and verification script
├── train_local.bat         # Windows training script
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory
│   └── train/
│       ├── cat/           # Cat images
│       ├── dog/           # Dog images
│       └── wild/          # Wild animal images
├── results_afhq/          # Training outputs
│   ├── model.pt           # Latest checkpoint
│   ├── model_*.pt         # Periodic checkpoints
│   └── sample_ddpm_*/     # Generated samples
└── wandb/                 # Training logs
```

## 🎯 Training Workflow

### 1. Initial Training

```bash
python main.py --data_class cat --train_steps 50000
```

### 2. Monitor Progress

- **Weights & Biases**: Check `./wandb/` directory for logs
- **Generated Samples**: Check `./results_afhq/sample_ddpm_*/` for generated images
- **Checkpoints**: Models saved in `./results_afhq/`

### 3. Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python main.py --load_path ./results_afhq/model.pt
```

### 4. Visualize Results

After training, visualize the diffusion process:

```bash
python main.py --visualize --load_path ./results_afhq/model.pt
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce memory usage
```bash
python main.py --batch_size 8 --image_size 64 --unet_dim 32
```

### Issue: Training is Very Slow

**Possible causes**:
1. **No GPU detected**: Run `python setup_local.py` to check GPU
2. **CPU training**: Make sure CUDA is installed correctly
3. **Too many workers**: Reduce `num_workers` in `trainer.py` (line 113)

### Issue: Dataset Not Found

**Solution**: 
1. Download dataset manually from: https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip
2. Extract to `./data/`
3. Ensure structure: `./data/train/cat/`, `./data/train/dog/`, `./data/train/wild/`

### Issue: wandb Login Prompt

The code is already configured for offline mode. If you still get prompts:

```bash
# Disable wandb completely
export WANDB_MODE=disabled  # Linux/Mac
set WANDB_MODE=disabled     # Windows CMD
```

Or in Python, add to `main.py`:
```python
os.environ['WANDB_MODE'] = 'disabled'
```

### Issue: Import Errors

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --upgrade
```

## 📈 Expected Training Times

Times are approximate and depend on your GPU:

| Configuration | GPU | Time per 1000 steps | Total (50k steps) |
|---------------|-----|---------------------|-------------------|
| Small (32x32) | RTX 3060 | ~2 min | ~1.5 hours |
| Medium (128x128) | RTX 3060 | ~8 min | ~6.5 hours |
| Large (256x256) | RTX 3090 | ~15 min | ~12.5 hours |

## 🎨 Sample Generation

After training, generate samples:

```bash
# Generate 512 samples
python main.py --load_path ./results_afhq/model.pt
```

Samples will be saved in `./results_afhq/sample_ddpm_*/`

## 📊 Monitoring Training

### Using Weights & Biases (W&B)

The training automatically logs to W&B in offline mode. To view logs:

1. Navigate to `./wandb/` directory
2. Find the latest run folder
3. Check `run-*.wandb` files

To sync with W&B cloud (optional):
```bash
wandb login
wandb sync ./wandb/offline-run-*
```

### Training Metrics

Key metrics to monitor:
- **Loss**: Should decrease over time
- **FID Score**: Lower is better (if `--fid` flag is used)
- **Generated Samples**: Visual quality improves over time

## 🎓 Training Tips

1. **Start Small**: Test with small parameters first to ensure everything works
2. **Use Checkpoints**: Training can take hours, save frequently
3. **Monitor GPU**: Use `nvidia-smi` to check GPU utilization
4. **Experiment**: Try different animal classes and parameters
5. **Be Patient**: Good results require 20k-50k steps minimum

## 🔍 Advanced Options

### Calculate FID Score

```bash
python main.py --fid --data_class cat
```

### Train on All Animals

```bash
python main.py --data_class all --train_steps 100000
```

### Custom Save Folder

```bash
python main.py --save_folder ./my_experiment --data_class dog
```

### Adjust Learning Rate

```bash
python main.py --learning_rate 0.0001
```

## 📝 Notes

- **Reproducibility**: Random seed is fixed in `main.py` for reproducible results
- **Data Augmentation**: Automatically applied during training
- **Multi-GPU**: Not currently supported, uses single GPU
- **Mixed Precision**: Not implemented, could be added for faster training

## 🆘 Getting Help

If you encounter issues:

1. Run `python setup_local.py` to diagnose problems
2. Check GPU availability: `nvidia-smi`
3. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check dataset structure: Ensure images are in correct folders

## 📚 Additional Resources

- **PyTorch CUDA Setup**: https://pytorch.org/get-started/locally/
- **AFHQ Dataset**: https://github.com/clovaai/stargan-v2
- **Diffusion Models**: https://arxiv.org/abs/2006.11239

---

**Happy Training! 🚀**
