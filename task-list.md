# CSCI-GA 2271 – Assignment 3 Task List

Step-by-step checklist to complete **Part A (required)** and **Part B (optional bonus)**.

---

## Part A – Variational Autoencoder (Required, 100 pts)

### 1. Setup

- [ ] Open the assignment folder and locate **`A3_PartA`**.
- [ ] Confirm the following files are in the same directory:
  - [ ] `variational_autoencoders.ipynb`
  - [ ] `vae.py`
  - [ ] Any helper files the notebook imports.
- [ ] Start your Python/Jupyter environment (local or Colab) and open **`variational_autoencoders.ipynb`**.

---

### 2. Understand the notebook & code structure

- [ ] Scroll through the notebook once (without running everything) and identify:
  - [ ] Where it imports `vae.py`.
  - [ ] Where the training loop is.
  - [ ] Where reconstructions / samples are visualized.
- [ ] Open **`vae.py`** in an editor and:
  - [ ] Find every `TODO` or placeholder to fill in.
  - [ ] Skim the class structure (e.g., `__init__`, `encode`, `reparameterize`, `decode`, `forward`, `loss`, etc., depending on template).

---

### 3. Implement the VAE in `vae.py`

- [ ] Implement the **encoder network**:
  - [ ] Layers that map input image → hidden features.
  - [ ] Heads that output **mu** and **logvar** (or equivalent) for the latent distribution.
- [ ] Implement the **reparameterization trick**:
  - [ ] Sample ε ~ N(0, I).
  - [ ] Compute `z = mu + std * eps`, where `std = exp(0.5 * logvar)` (or as specified).
- [ ] Implement the **decoder network**:
  - [ ] Layers that map latent `z` → reconstructed image `x_recon`.
- [ ] Implement the **loss function**:
  - [ ] Reconstruction loss: compare `x_recon` with original `x` (BCE or MSE, per instructions).
  - [ ] KL divergence term: KL(q(z|x) || N(0, I)).
  - [ ] Return total loss (e.g., `recon_loss + kl_loss`).

---

### 4. Run and test the VAE notebook

- [ ] In `variational_autoencoders.ipynb`, run the initial setup cells (imports, dataset loading, etc.).
- [ ] Run the cell that **instantiates the VAE model**:
  - [ ] Confirm no shape mismatch or device errors.
- [ ] Run the **training loop**:
  - [ ] Confirm training runs without crashing.
  - [ ] Observe that loss decreases over epochs (at least roughly).
- [ ] Run the **evaluation / visualization cells**:
  - [ ] Show reconstructed images.
  - [ ] Show samples from the latent space (if provided).
- [ ] Fill in any **inline text answers** in the notebook, if present (e.g., conceptual questions about KL term, latent space, etc.).

---

### 5. Clean up Part A & prepare for submission

- [ ] Restart the notebook kernel and **Run All** to ensure it works from a clean start.
- [ ] Save **`variational_autoencoders.ipynb`** with final outputs.
- [ ] Confirm the only Part A code files you modified are:
  - [ ] `vae.py`
  - [ ] `variational_autoencoders.ipynb`
- [ ] Add these Part A files to a folder ready for inclusion in the final submission zip.

---

## Part B – Diffusion Models (Optional Bonus, +50 pts)

> Only do this if you have time + GPU resources.

### 6. Environment & data setup (Part B)

- [ ] Open **`A3_PartB`** folder and confirm it contains:
  - [ ] `diffusion.py`
  - [ ] `unet.py`
  - [ ] `trainer.py`
  - [ ] `utils.py`
  - [ ] `run_diffusion.ipynb`
  - [ ] `requirements.txt`
  - [ ] `data/` (or plan to create/populate it).
- [ ] Decide where to run:
  - [ ] Google Colab / Kaggle / cloud VM with GPU.
- [ ] Open **`run_diffusion.ipynb`** in that environment.
- [ ] Install dependencies (e.g., using `pip install -r requirements.txt` or commands provided in the notebook).
- [ ] Download and prepare the AFHQ (cats) dataset:
  - [ ] Create `./data` directory.
  - [ ] Use `gdown` (or provided command) to download `afhq_v2.zip` into `./data`.
  - [ ] Unzip into `./data`.
- [ ] Verify that the notebook can **load the dataset** without errors.

---

### 7. Understand the diffusion code

- [ ] Open **`diffusion.py`** and:
  - [ ] Locate all `TODO` markers.
  - [ ] Identify which functions you must implement:
    - [ ] `__init__`
    - [ ] `forward`
    - [ ] `q_sample`
    - [ ] `p_loss`
    - [ ] `sample`
    - [ ] `p_sample`
    - [ ] `p_sample_loop`
- [ ] Skim:
  - [ ] `trainer.py` to see how `Diffusion` is used in training.
  - [ ] `unet.py` to understand the ε_θ U-Net architecture.
  - [ ] `utils.py` for helpers (e.g., `extract`, logging utilities).
  - [ ] `run_diffusion.ipynb` to see how training and visualization functions are called.

---

### 8. Implement the forward process & loss in `Diffusion`

- [ ] In `__init__`:
  - [ ] Implement the **cosine noise schedule** to compute ᾱ_t.
  - [ ] Derive α_t from ᾱ_t.
  - [ ] Precompute and store tensors (α_t, ᾱ_t, etc.) on the correct device.
- [ ] Implement `q_sample`:
  - [ ] Given x₀ and t, compute `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps`.
  - [ ] Use `extract` to broadcast coefficients to match batch shape.
- [ ] Implement `forward`:
  - [ ] Sample t uniformly from {1,…,T}.
  - [ ] Sample ε ~ N(0, I).
  - [ ] Compute x_t = q_sample(x₀, t, ε).
  - [ ] Return (x_t, t, ε) (or as required by the template).
- [ ] Implement `p_loss`:
  - [ ] Use `forward` (or replicate its logic) to get x_t, t, ε.
  - [ ] Pass (x_t, t) into the U-Net ε_θ to get predicted noise.
  - [ ] Compute **L1 loss** between predicted noise and true ε.
  - [ ] Return this loss for the optimizer.

---

### 9. Implement reverse process & sampling

- [ ] Implement `p_sample` (single reverse step):
  - [ ] Given x_t and t, compute ε_θ(x_t, t).
  - [ ] Compute x̂₀ using the formula derived from Eq. (2) (rearranged).
  - [ ] Clamp x̂₀ to [-1, 1].
  - [ ] Compute μ̃_t and σ_t from the closed-form posterior (Eq. (7)).
  - [ ] Sample z ~ N(0, I) if t > 1, else set z = 0.
  - [ ] Compute x_{t−1} = μ̃_t + σ_t * z.
- [ ] Implement `p_sample_loop`:
  - [ ] Sample x_T ~ N(0, I).
  - [ ] For t = T down to 1:
    - [ ] Update x_t → x_{t−1} using `p_sample`.
    - [ ] Optionally store intermediate x_t’s for visualization.
  - [ ] Return final x₀ (and intermediates if needed).
- [ ] Implement `sample`:
  - [ ] Call `p_sample_loop` for a batch of samples.
  - [ ] Reshape/rescale outputs to image format expected by visualization code.

---

### 10. Train and check basic behavior (Part 1)

- [ ] In `run_diffusion.ipynb`, configure **Part 1** parameters:
  - [ ] `train_steps = 1000`
  - [ ] `save_and_sample_every = 100`
  - [ ] `fid = False`
- [ ] Run the training cell:
  - [ ] Confirm training completes without errors.
  - [ ] Confirm training loss is logged.
  - [ ] Confirm sample images are periodically generated (blurry cats).
- [ ] Produce a **training loss vs steps plot** (Part 1 requirement).

---

### 11. Train with FID and plot it (Part 2)

- [ ] Modify config for **Part 2**:
  - [ ] `train_steps = 1000`
  - [ ] `save_and_sample_every = 100`
  - [ ] `fid = True`
- [ ] Run training again:
  - [ ] Confirm FID is computed every 100 steps.
- [ ] Extract FID values from logs.
- [ ] Create and save a **FID vs training steps plot**.

---

### 12. Visualize forward & backward diffusion (Parts 3 & 4)

**Forward diffusion (Part 3):**

- [ ] With the model trained for 1000 steps:
  - [ ] Take the first batch x₀ from the training data.
  - [ ] Use provided visualization helper (e.g., `visualize_diffusion`) to show x_t at:
    - [ ] 0% of total timesteps
    - [ ] 25%
    - [ ] 50%
    - [ ] 75%
    - [ ] 99%
- [ ] Confirm images gradually get more noisy as t increases.

**Backward diffusion (Part 4):**

- [ ] Start from the noisy images at the last forward timestep (x_T).
- [ ] Run your reverse diffusion (sampling) and capture intermediate outputs at:
  - [ ] 0%
  - [ ] 25%
  - [ ] 50%
  - [ ] 75%
  - [ ] 99% of reverse steps.
- [ ] Confirm images gradually denoise into cat images.

> Parts 5 & 6 (10,000 steps) are **optional / ungraded**; only run them if you want additional practice and better image quality.

---

## 13. Final Submission Prep

- [ ] For **Part A**, include:
  - [ ] `variational_autoencoders.ipynb`
  - [ ] `vae.py`
- [ ] For **Part B** (if completed), include:
  - [ ] `diffusion.py`
  - [ ] `run_diffusion.ipynb`
  - [ ] Any other `.py` files you actually modified (e.g., `trainer.py` if changed).
- [ ] For each notebook you’re submitting:
  - [ ] Restart kernel and **Run All** to ensure no missing imports or path issues.
- [ ] Put all modified `.py` and `.ipynb` files into **one zip file**.
- [ ] Upload the zip to the course submission portal (Gradescope / Brightspace / etc.).

---
