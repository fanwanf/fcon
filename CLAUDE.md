# FCON Project - Claude Instructions

## CRITICAL: AWS Credential Refresh (MANDATORY)

**AWS credentials expire every 30 minutes. You MUST maintain a running credential refresh process for the entire duration of any long-running job.**

### Credential refresh command:
```bash
ada credentials update --provider=conduit --account=195966524180 --role=IibsAdminAccess-DO-NOT-DELETE --once
```

### At the start of EVERY session or long-running task:

1. **Start the background refresh loop immediately:**
   ```bash
   /home/fanwanf/lab42/fcon/refresh_creds.sh > /tmp/creds_refresh.log 2>&1 &
   echo $! > /tmp/creds_refresh.pid
   ```

2. **Verify it is running:**
   ```bash
   cat /tmp/creds_refresh.log
   ```

3. **The script refreshes every 25 minutes** (5-minute safety margin before the 30-minute expiry).

4. **If credentials appear stale or a job fails with auth errors**, re-run the refresh command manually and restart the loop.

5. **Check if the loop is still alive** during long jobs:
   ```bash
   kill -0 $(cat /tmp/creds_refresh.pid) 2>/dev/null && echo "running" || echo "DEAD - restart it!"
   ```

The refresh script lives at `/home/fanwanf/lab42/fcon/refresh_creds.sh`.

---

## Project Overview

- **Model**: F-CON (Frustum Convolutional Occupancy Network) for 3D shape completion
- **Paper**: "Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects" (IROS 2023)
- **Dataset**: COB-3D-v2 at `/tmp/cob3d/v2`
- **Checkpoint**: `checkpoints/fcon.pt`

## Environment

- **Conda env**: `fcon` at `/home/fanwanf/miniconda3/envs/fcon`
- **Python**: `/home/fanwanf/miniconda3/envs/fcon/bin/python`
- **GPUs**: 8x NVIDIA L4 (23 GB each)

## Running Evaluation

```bash
# Single GPU
/home/fanwanf/miniconda3/envs/fcon/bin/python evaluate.py \
    --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt

# Multi-GPU (recommended)
/home/fanwanf/miniconda3/envs/fcon/bin/python evaluate.py \
    --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt --n_workers 8
```

Logs go to `/tmp/fcon_eval.log` when run in background.

## Expected Results (Table I of paper)

| Metric   | Paper value |
|----------|-------------|
| CD-L1    | 0.4229      |
| CD-L2    | 0.1157      |
| F1@0.1   | 0.4664      |
| F1@0.3   | 0.8928      |
| F1@0.5   | 0.9600      |
| Box-IoU  | 0.6809      |
| Box-IoG  | 0.7686      |
| Box-F1   | 0.7485      |

---

## Goal: Improve Occupancy Estimation with Scene Context & Background Objects

The core problem: F-CON predicts each object's 3D shape **independently**, using only the object's local frustum (cropped RGB + segmentation mask). This causes severe depth errors — e.g., boxes on a conveyor belt are predicted as floating shells because the model has no information about:
- Where the conveyor/support surface is
- Other boxes providing size/shape priors
- Surface orientation normals constraining depth extent

---

## Architecture Deep-Dive: How Depth is Currently Predicted

### Pipeline (fcon_model.py)

1. **Frustum definition** (`_get_frustums`):
   - near_plane = min depth of visible object pixels (from depth_map masked by segm)
   - far_plane = global scene far_plane
   - ROI = 2D bounding box enlarged by 1.2×

2. **Voxelization** (`_voxelize_roi_point_clouds`):
   - For each pixel in the 64×64 ROI, find which of 96 depth bins its actual depth falls in
   - Creates a (N_obj, 96, 64, 64) one-hot binary tensor — only foreground object pixels
   - **KEY LIMITATION**: Background depth (conveyor, table) is silently discarded

3. **ROI feature construction** (`_compute_roi_features`):
   - `roi_feats = cat([rgb_p, masks_p, masks_p], dim=1)` → 5 channels: RGB + mask (duplicated)
   - **KEY LIMITATION**: mask is duplicated (not mask + amodal_mask); normal_map is NEVER used
   - Input volume: `einsum("bdhw,bchw->bcdhw", voxelized_points, roi_feats)` → (N, 5, 96, 64, 64)
   - Only voxels at the measured depth bin are non-zero; everything else is 0

4. **3D-UNet** (channels 32→64→128→256→32→1):
   - Processes the sparse input volume to predict occupancy in the full frustum
   - All context is local to the per-object frustum — no other objects, no scene background

5. **Squeeze + 2D-UNet**:
   - Collapses 3D→2D: (N, 1, 96, 64, 64) → (N, 96, 64, 64), treating depth bins as channels
   - 2D-UNet refines spatial features across (H, W) for each depth slice
   - **Depth prediction is essentially learned from the single visible surface** at near_plane

6. **Predictor** → (N, 96, 64, 64) logits = per-voxel occupancy in frustum

### What the model sees vs. what it needs

| Information | Currently used | Available in dataset |
|---|---|---|
| Object RGB crop | ✅ | `rgb` ROI-aligned |
| Object visible mask | ✅ (doubled) | `segm.masks` |
| Object amodal mask | ❌ | `segm.amodal_masks` |
| Surface normals | ❌ | `normal_map` (3×H×W) |
| Background depth in ROI | ❌ | `depth_map` (full scene) |
| Global scene RGB features | ❌ | `rgb` (full image) |
| Support surface position | ❌ | derivable from `depth_map` |
| Other objects' poses/shapes | ❌ | `obj_poses`, other frustums |

---

## TODO List (Ordered by Impact / Effort Ratio)

### TIER 1 — Quick wins, high signal-to-noise, implement first

#### TODO-1: Add surface normals as input (amodal_mask bonus)
**File**: `fcon_model.py` (`_compute_roi_features`), `train.py`, `cob3d_dataset.py`
**Effort**: 1 day | **Expected gain**: medium-high (normals directly constrain depth direction)

- ROI-align `normal_map` (3 channels) into the 64×64 patch grid
- ROI-align `amodal_masks` (1 channel) as a second mask channel
- Replace `cat([rgb_p, masks_p, masks_p])` with `cat([rgb_p, masks_p, amodal_masks_p, normals_p])` → 8 channels
- Change first ConvLayer in `feature_module` from `5 →` to `8 →` input channels
- Add `normal_map` and `amodal_masks` to `train_step` and `val_step` data flow
- **Why**: A face with up-pointing normals is a horizontal surface → the box extends downward. A forward-facing normal means the visible face is the front → object extends backward by its depth. Directly constrains shape completion.

#### TODO-2: Include background depth voxelization as a second input volume
**File**: `fcon_model.py` (`_voxelize_roi_point_clouds`, `_compute_roi_features`)
**Effort**: 1-2 days | **Expected gain**: high (support surface explicitly visible)

- Current: `voxelized_roi_point_clouds` is (N, 96, H, W), only foreground pixels (masked by segm.masks)
- Add `voxelized_roi_background` = same but for pixels NOT in the mask (background, conveyor)
- Combine as a 2-channel volume: `stack([foreground_voxels, background_voxels], dim=1)` → (N, 2, 96, H, W)
- Einsum: object features → (N, 5, 96, H, W), background depth → (N, 1, 96, H, W) → concatenate → (N, 6, 96, H, W)
- **Why**: The conveyor belt pixels appear at a specific depth in the background. This makes "the box sits on the belt at depth Z" explicit. The model can then reliably extend the box from its visible top surface down to Z.

---

### TIER 2 — Global context, moderate effort, highest architectural impact

#### TODO-3: Global scene encoder with per-object feature injection
**New file**: `scene_encoder.py` | **Modified**: `fcon_model.py`, `train.py`
**Effort**: 3-5 days | **Expected gain**: very high (core fix for the stated problem)

Architecture:
- Add a lightweight 2D CNN encoder (e.g., 4-layer FPN-style) that processes the full RGB+depth image: (4, H, W) → multi-scale feature maps
- For each object, ROI-align the global feature map into (C_global, 64, 64)
- Inject into the 2D-UNet (after 3D→2D collapse) via concatenation or FiLM conditioning
- The object's 2D-UNet now sees: local depth-binned features + what the rest of the scene looks like

Specifically:
```
global_feats = GlobalEncoder(rgb, depth_map)          # (C_g, H, W)
per_obj_ctx  = roi_align(global_feats, boxes)          # (N, C_g, 64, 64)
# After 3D-UNet squeeze:
feats_2d = cat([feats_2d, per_obj_ctx], dim=1)        # inject before 2D-UNet
```

**Why**: The global encoder "sees" the conveyor belt, wall, table, and all objects. When processing each individual object, it can now answer "is there a surface below me?" and "how far do similar-looking boxes nearby extend in depth?"

#### TODO-4: Support surface detection and conditioning
**New file**: `support_plane.py` | **Modified**: `fcon_model.py`, `train.py`
**Effort**: 2-3 days | **Expected gain**: high for bin-packing/conveyor scenes

- Use RANSAC on the full depth_map to fit the dominant ground/support plane
- Encode plane as (distance_from_near, normal_vector) = 4 floats
- Inject as a global conditioning vector via FiLM layers in the 3D-UNet
- **Why**: The most reliable physical constraint for any object on a conveyor is "it touches the belt". This single plane equation anchors the depth prediction from top-surface to support.

---

### TIER 3 — Cross-object attention (high complexity, highest long-term impact)

#### TODO-5: Transformer cross-attention between objects in the same scene
**Modified**: `fcon_model.py` `forward()`, `train.py`
**Effort**: 5-7 days | **Expected gain**: very high for multi-object scenes

Architecture:
- After 3D-UNet squeeze → (N_obj, 96, 64, 64) features per scene
- Global average pool each object's feature to a (N_obj, C) token
- Apply multi-head self-attention across all N_obj tokens (or cross-attention to a scene token)
- Broadcast attended features back to (N_obj, C', 64, 64) via spatial broadcast
- Feed into 2D-UNet

**Why**: All boxes in a scene tend to have similar depths. With attention, one well-constrained object (e.g., an isolated box with clear depth) can inform depth predictions for nearby occluded boxes. Also catches geometric regularity: boxes arranged in a grid all have the same floor depth.

---

### TIER 1 — Confirmed fixes from training + evaluation (implement before next retrain)

#### TODO-6: Increase resolution — n_depth_bins 96→192, patch_size 64→96 ✅ CONFIRMED NEEDED
**File**: `fcon_model.py`, `train.py`, `fcon_model_v1.py`
**Effort**: ~1 hour (config change) | **Expected gain**: high (current res ~3-6mm, target ~2mm)

- Change `n_depth_bins=96` → `192` in model construction and all callers
- Change `patch_size=64` → `96` in model construction and all callers
- First ConvLayer in `feature_module` input channels unchanged (15 for v2, 5 for v1)
- **Memory impact**: 3D feature volume grows from (N,15,96,64,64) → (N,15,192,96,96), roughly 6× larger
  - With obj_chunk=32 and AMP fp16, this is ~18 GB per GPU on L4 — may need to reduce obj_chunk to 16
- **Why confirmed**: Current depth bin width = frustum_range/96 ≈ 3-5mm. Spatial patch = object_size/64 ≈ 2-6mm.
  Both are too coarse for 2mm accuracy targets. Doubling both dims brings worst-case to ~2mm.

#### TODO-7: Depth range randomization during training ✅ CONFIRMED NEEDED
**File**: `train.py` (`train_step`), `fcon_model.py` (`_get_frustums`)
**Effort**: ~2 hours | **Expected gain**: high (prevents frustum-range "cheat")

**The bug found**: `far_plane` in COB3D is set to ~floor/conveyor depth (within 1-8cm). This means
even without background channels, the old model can implicitly read box height from `far_plane - near_plane`.
Ablation on synthetic data confirmed: old model CD-L2 degraded +193% with randomized far_plane vs +132% for new model.

**Fix**: During training, add random cushion to near and far planes per-object:
```python
# In train_step, before calling model.predict():
cushion_top    = torch.rand(n_obj, device=dev).mul(0.30)  # 0–30cm above box top
cushion_bottom = torch.rand(n_obj, device=dev).mul(0.50)  # 0–50cm past floor
# Pass these as near_plane_offset / far_plane_offset args into _get_frustums()
# (Currently _get_frustums uses a fixed 5cm near offset; make it a per-object random tensor)
```
At inference time, keep the original fixed 5cm near offset and actual scene far_plane.
This forces the model to infer box depth from background/normal evidence, not frustum range.

---

### TIER 1 — Selected for production bin-packing accuracy

#### TODO-8: Focal loss for voxel class imbalance ✅ IMPLEMENTED
**File**: `train.py` (`train_step`)
**Effort**: done | **Status**: in production code

- ~95% of frustum voxels are empty → BCE biases the model toward predicting empty everywhere
- Focal loss `FL = -(1-p_t)^γ * BCE`, γ=2: down-weights easy empty-voxel predictions, improving occupied recall
- Applied with the existing `pos_weight` so positive-class imbalance is handled at both levels
- **Why for bin-packing**: under-filled predictions leave gaps in the predicted box interior → robot assumes free space that is actually occupied → collision during placement

#### TODO-9: Largest connected component filter ✅ IMPLEMENTED
**File**: `evaluate.py` (`evaluate_scene`)
**Effort**: done | **Status**: in production code

- After `probs = logits.sigmoid()`, binarize at 0.5 and run `scipy.ndimage.label` on each object's 3D volume
- Keep only the single largest connected blob; set all other voxels to 0 before marching cubes
- No retraining required — pure post-processing
- **Why for bin-packing**: floating voxel fragments produce spurious predicted surfaces that inflate bounding-box estimates and cause false collision detections

#### TODO-10: Separate near/far surface heads (replace per-voxel binary occupancy)
**File**: `fcon_model.py` (new output heads), `train.py` (new loss)
**Effort**: 3-5 days | **Expected gain**: highest for box accuracy (solid output by construction)

- Current: per-voxel binary occupancy (N, D, H, W) → marching cubes
- Replace predictor with two heads per spatial location (H, W):
  - `near_surface_head` → predicted depth bin of the top/front surface (softmax over D)
  - `far_surface_head`  → predicted depth bin of the bottom/back surface (softmax over D)
- Occupancy is then filled solid between the two predicted surfaces — boxes are **always** connected by construction
- Train with cross-entropy loss against GT surface bin indices (derived from GT occupancy boundary)
- **Why for bin-packing**: eliminates floating voxels and interior holes architecturally, not via post-processing. The output is always a valid solid box shape.

---

## Implementation Order (recommended)

```
TODO-9 (LCC filter)        →  no retrain, already in evaluate.py
TODO-7 (depth range rand)  →  enhanced perturbation, already in fcon_model.py
TODO-8 (focal loss)        →  already in train.py

TODO-1 + TODO-2 + TODO-7 + TODO-8 combined  →  retrain 125 epochs  →  evaluate Box-IoU/F1
TODO-4 (support plane)     →  add support_plane.py + FiLM conditioning  →  retrain  →  evaluate
TODO-10 (near/far heads)   →  replace predictor head  →  retrain  →  evaluate
TODO-3 (global encoder)    →  add scene_encoder.py + ROI inject  →  retrain  →  evaluate
```

Run each ablation using `evaluate.py --n_scenes 100` for fast iteration before full 696-scene eval.

---

## Key Files Reference

| File | Role |
|---|---|
| `fcon_model.py` | Model: frustum setup, voxelization, 3D-UNet, 2D-UNet, predictor |
| `train.py` | Training loop: AMP fp16, 8-GPU DataParallel, obj_chunk=32 |
| `evaluate.py` | Metrics: CD-L1/L2, F1@{0.1,0.3,0.5}, Box-IoU/IoG/F1 |
| `frustum.py` | Frustum geometry: near/far planes, to_grid() for 3D point generation |
| `cob3d_dataset.py` | Dataset: loads rgb, depth, normals, masks, voxels, poses |
| `layers/unet.py` | UNet implementation (2D and 3D, shared) |
| `torch_utils.py` | depth2cloud, interp3d, transform_points, enlarge_box_roi |

## Training Commands

```bash
# Full 125-epoch training with all fixes (TODO-6 + TODO-7 + TODO-1 + TODO-2)
nohup /home/fanwanf/miniconda3/envs/fcon/bin/python train.py \
    --data_root /tmp/cob3d/v2 --epochs 125 --lr 1e-3 \
    --checkpoint checkpoints/fcon_v2.pt > /tmp/train.log 2>&1 &

# Resume from checkpoint
nohup /home/fanwanf/miniconda3/envs/fcon/bin/python train.py \
    --data_root /tmp/cob3d/v2 --epochs 125 --lr 1e-3 \
    --resume checkpoints/fcon_v2_best.pt --checkpoint checkpoints/fcon_v2.pt > /tmp/train.log 2>&1 &
```

