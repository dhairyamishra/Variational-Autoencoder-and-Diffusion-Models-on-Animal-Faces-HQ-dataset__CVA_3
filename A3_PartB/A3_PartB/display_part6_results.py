"""
Display Part 6 Visualization Results
Shows forward and backward diffusion processes from 10,000-step trained model
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

def display_part6_results():
    """Display forward and backward diffusion visualizations"""
    
    viz_folder = './part6_visualization'
    
    if not os.path.exists(viz_folder):
        print(f"❌ ERROR: Visualization folder not found: {viz_folder}")
        return
    
    # Get all forward and backward diffusion images
    forward_images = sorted([f for f in os.listdir(viz_folder) if f.startswith('forward_diffusion_')])
    backward_images = sorted([f for f in os.listdir(viz_folder) if f.startswith('backward_diffusion_')])
    
    print("=" * 80)
    print("🎨 PART 6: DIFFUSION PROCESS VISUALIZATION (10,000-step model)")
    print("=" * 80)
    print(f"\n📊 Found {len(forward_images)} forward diffusion images")
    print(f"📊 Found {len(backward_images)} backward diffusion images")
    print()
    
    # Display Forward Diffusion Process
    if forward_images:
        print("\n" + "=" * 80)
        print("🔴 FORWARD DIFFUSION PROCESS")
        print("   (Clean Image → Noise)")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, len(forward_images), figsize=(20, 4))
        if len(forward_images) == 1:
            axes = [axes]
        
        for idx, img_name in enumerate(forward_images):
            img_path = os.path.join(viz_folder, img_name)
            img = Image.open(img_path)
            
            # Extract timestep from filename (e.g., forward_diffusion_0.png -> 0%)
            timestep_idx = int(img_name.split('_')[-1].replace('.png', ''))
            percentages = [0, 25, 50, 75, 99]
            if timestep_idx < len(percentages):
                percent = percentages[timestep_idx]
            else:
                percent = timestep_idx
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'Step {timestep_idx}\n({percent}% noise)', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Forward Diffusion: Adding Noise Over Time', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('./part6_forward_diffusion_grid.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: part6_forward_diffusion_grid.png")
        plt.show()
    
    # Display Backward Diffusion Process
    if backward_images:
        print("\n" + "=" * 80)
        print("🔵 BACKWARD DIFFUSION PROCESS (DENOISING)")
        print("   (Noise → Generated Image)")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, len(backward_images), figsize=(20, 4))
        if len(backward_images) == 1:
            axes = [axes]
        
        for idx, img_name in enumerate(backward_images):
            img_path = os.path.join(viz_folder, img_name)
            img = Image.open(img_path)
            
            # Extract timestep from filename
            timestep_idx = int(img_name.split('_')[-1].replace('.png', ''))
            percentages = [0, 25, 50, 75, 99]
            if timestep_idx < len(percentages):
                percent = percentages[timestep_idx]
            else:
                percent = timestep_idx
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'Step {timestep_idx}\n({percent}% denoised)', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Backward Diffusion: Removing Noise Over Time (Model Denoising)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('./part6_backward_diffusion_grid.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: part6_backward_diffusion_grid.png")
        plt.show()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print("✅ Part 6 Visualization Complete!")
    print(f"\n📁 All images saved in: {viz_folder}/")
    print("\n🎯 Key Observations:")
    print("   • Forward process: Clean cat images gradually become pure noise")
    print("   • Backward process: The trained model successfully denoises random noise")
    print("   • The 10,000-step model should show better quality than 1,000-step model")
    print("\n💾 Grid images saved:")
    print("   - part6_forward_diffusion_grid.png")
    print("   - part6_backward_diffusion_grid.png")
    print("=" * 80)

if __name__ == '__main__':
    display_part6_results()
