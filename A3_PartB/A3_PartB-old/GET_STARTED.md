# 🚀 Get Started - 3 Simple Steps

Welcome! This guide will get you training your diffusion model on your local GPU in just 3 steps.

## Step 1: Install Dependencies (5 minutes)

Open a terminal in this directory and run:

```bash
# Install PyTorch with CUDA support (choose your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Step 2: Setup & Verify (2 minutes)

Run the setup script to check your GPU and download the dataset:

```bash
python setup_local.py
```

This will:
- ✅ Check if your GPU is working
- ✅ Verify all dependencies
- ✅ Help you download the AFHQ dataset (6.5GB)
- ✅ Show you example training commands

## Step 3: Start Training!

### Option A: Quick Test First (Recommended)

Verify everything works with a 1-minute test:

```bash
python quick_test.py
```

### Option B: Start Full Training

**Easiest way** - Just double-click:
```
train_local.bat
```

**Or use command line:**
```bash
# Standard training (recommended for most GPUs)
python main.py --data_class cat --train_steps 50000

# Quick test (small model, fast)
python main.py --time_steps 50 --train_steps 1000 --batch_size 16 --image_size 32 --unet_dim 16 --data_class cat

# High quality (needs 16GB+ GPU)
python main.py --time_steps 1000 --train_steps 100000 --batch_size 16 --image_size 256 --unet_dim 128 --data_class cat
```

---

## 📊 What to Expect

- **Training Time**: 6-12 hours for 50k steps (depends on GPU)
- **GPU Memory**: 4-8GB for standard settings
- **Output**: Generated images in `./results_afhq/sample_ddpm_*/`
- **Checkpoints**: Saved every 1000 steps in `./results_afhq/`

## 🔧 Common Issues

### "CUDA out of memory"
Reduce batch size and image size:
```bash
python main.py --batch_size 8 --image_size 64 --unet_dim 32
```

### "Dataset not found"
Download manually:
1. Get from: https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip
2. Extract to `./data/`
3. Ensure structure: `./data/train/cat/`, `./data/train/dog/`, `./data/train/wild/`

### "No GPU detected"
Check CUDA installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 Need More Help?

- **Full Documentation**: See `README_LOCAL.md`
- **GPU Settings**: Adjust parameters in `train_local.bat`
- **Resume Training**: `python main.py --load_path ./results_afhq/model.pt`

---

## 🎯 Quick Reference

| File | Purpose |
|------|---------|
| `setup_local.py` | Check GPU, download dataset, verify setup |
| `quick_test.py` | Run 1-minute test to verify everything works |
| `train_local.bat` | Easy Windows training script |
| `main.py` | Main training script with all options |
| `README_LOCAL.md` | Complete documentation |

---

**Ready to train? Run `python quick_test.py` to verify your setup!** ✨
