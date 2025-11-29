# Diffusion Model Assignment Task List

This task list outlines the steps to complete the diffusion model assignment. We will work through each part one by one, checking them off as completed.

- [ ] **Part 1 (10 pts)**: Train diffusion model for 1000 steps with parameters `train_steps=1000`, `save_and_sample_every=100`, `fid=False`. Expected runtime: 5-10 minutes. Generate training loss plot.
- [ ] **Part 2 (20 pts)**: Train diffusion model for 1000 steps with parameters `train_steps=1000`, `save_and_sample_every=100`, `fid=True`. Expected runtime: 15-60 minutes. Compute and plot FID over training steps.
- [ ] **Part 3 (10 pts)**: Using model trained for 1000 steps, visualize forward diffusion process at 0%, 25%, 50%, 75%, 99% of timesteps on initial batch.
- [ ] **Part 4 (10 pts)**: Using model trained for 1000 steps, visualize backward diffusion process at 0%, 25%, 50%, 75%, 99% of timesteps starting from noise of last timestep.
- [ ] **Part 5 (Ungraded Exercise)**: Train diffusion model for 10000 steps with parameters `train_steps=10000`, `save_and_sample_every=1000`, `fid=False`. Expected runtime: 2 hours. Show final generated images.
- [ ] **Part 6 (Ungraded Exercise)**: Using model trained for 10000 steps, repeat Part 3 and Part 4 visualizations.
