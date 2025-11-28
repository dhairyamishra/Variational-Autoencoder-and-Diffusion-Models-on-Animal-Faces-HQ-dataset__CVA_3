# PART 2: Plot Training Loss from Wandb Run
# Copy this into a Jupyter notebook cell

import wandb
import matplotlib.pyplot as plt
import os

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
    plt.plot(steps, losses, linewidth=2, color='#2E86AB', marker='o', markersize=3)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Diffusion Model Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Plot saved to: training_loss.png")
    
except Exception as e:
    print(f"⚠️ Could not load from Wandb cloud: {e}")
    print("\n💡 Solution: Sync your run to Wandb cloud first:")
    print(f"   !wandb sync {run_path}")
    print("\n   Then run this cell again!")
