# PART 2: Plot Training Loss AND Display Sample Images
# Copy this into a Jupyter notebook cell

import wandb
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# ============================================
# PART A: Plot Training Loss
# ============================================

# Find the most recent wandb run
wandb_dir = './wandb'
runs = [d for d in os.listdir(wandb_dir) if d.startswith('run-')]
run_path = os.path.join(wandb_dir, sorted(runs)[-1])
run_id = os.path.basename(run_path).split('-')[-1]

print(f"📊 Reading run: {run_id}")

# Initialize Wandb API and get the run
api = wandb.Api()

try:
    # Try to get run from Wandb cloud (requires sync)
    run = api.run(f"DDPM_AFHQ/{run_id}")
    history = run.history()
    
    # Extract loss values
    steps = history['_step'].values
    losses = history['loss'].values
    
    print(f"✅ Found {len(losses)} loss values")
    print(f"   Loss: {losses.min():.6f} → {losses.max():.6f}")
    
    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=2, color='#2E86AB')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Diffusion Model Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Loss plot saved to: training_loss.png")
    
except Exception as e:
    print(f"⚠️ Could not load from Wandb cloud: {e}")
    print("\n💡 Solution: Sync your run to Wandb cloud first:")
    print(f"   !wandb sync {run_path}")


# ============================================
# PART B: Display Generated Sample Images
# ============================================

print("\n" + "="*60)
print("📸 Looking for generated sample images...")

# Find the most recent sample image
results_folder = './results'
sample_files = [f for f in os.listdir(results_folder) if f.startswith('sample_') and f.endswith('.png')]

if sample_files:
    # Get the most recent sample
    latest_sample = sorted(sample_files)[-1]
    sample_path = os.path.join(results_folder, latest_sample)
    
    print(f"✅ Found sample: {latest_sample}")
    
    # Display the sample image
    img = Image.open(sample_path)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Generated Samples: {latest_sample}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Sample image saved to: generated_samples.png")
    print(f"   Image size: {img.size}")
    
else:
    print("⚠️ No sample images found in ./results/")


# ============================================
# PART C: Display Sample Folders (if available)
# ============================================

# Check for sample_ddpm folders
sample_folders = sorted([d for d in os.listdir(results_folder) if d.startswith('sample_ddpm_')])

if sample_folders:
    print(f"\n✅ Found {len(sample_folders)} sample folders")
    print(f"   Latest: {sample_folders[-1]}")
    
    # Display a few samples from the latest folder
    latest_folder = os.path.join(results_folder, sample_folders[-1])
    sample_images = sorted([f for f in os.listdir(latest_folder) if f.endswith('.png')])[:16]
    
    if sample_images:
        print(f"   Displaying first 16 samples from {sample_folders[-1]}...")
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Individual Samples from {sample_folders[-1]}', fontsize=14, fontweight='bold')
        
        for idx, (ax, img_name) in enumerate(zip(axes.flat, sample_images)):
            img_path = os.path.join(latest_folder, img_name)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Sample {idx}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('individual_samples_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Individual samples grid saved to: individual_samples_grid.png")

print("\n" + "="*60)
print("🎉 All done!")
