"""
Training and validation script for F-CON.

Usage:
    # Train from scratch (paper config):
    python train.py --data_root /tmp/cob3d/v2 --epochs 125 --lr 1e-3

    # Resume:
    python train.py --data_root /tmp/cob3d/v2 --resume checkpoints/fcon.pt --epochs 125 --lr 1e-3

    # Validate only:
    python train.py --data_root /tmp/cob3d/v2 --resume checkpoints/fcon.pt --val_only
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cob3d_dataset import COB3D
from fcon_model import FCON
from torch_utils import depth2cloud, to_torch, transform_points


# ---------------------------------------------------------------------------
# GT occupancy helper
# ---------------------------------------------------------------------------

def compute_gt_occupancy(grid_centers, cam_from_obj, scales, extents, voxels):
    """Project frustum voxel centers into object space and sample GT occupancy."""
    obj_from_cam = torch.inverse(cam_from_obj)
    pts_obj = transform_points(obj_from_cam[:, None, None, None], grid_centers)
    half_extent = (extents / scales).view(-1, 1, 1, 1, 3)
    pts_normed = pts_obj / half_extent
    gt_occ = F.grid_sample(
        voxels[:, None].float(),
        pts_normed.flip(-1),
        mode="nearest",
        align_corners=True,
        padding_mode="zeros",
    ).squeeze(1)
    return gt_occ


# ---------------------------------------------------------------------------
# Dataset wrapper + collate
# ---------------------------------------------------------------------------

class SceneDataset(Dataset):
    def __init__(self, dset, scene_ids):
        self.dset = dset
        self.scene_ids = list(scene_ids)

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        return to_torch(self.dset[self.scene_ids[idx]], recursive=True)


def list_collate(batch):
    """Return scenes as a list — each scene has a different number of objects."""
    return batch


# ---------------------------------------------------------------------------
# Training step: one scene per optimizer step with AMP
# ---------------------------------------------------------------------------

def train_step(model, model_dp, batch, device, optimizer, scaler, obj_chunk=32):
    """Process one scene, chunking objects to bound peak GPU memory."""
    rgb          = batch["rgb"].to(device)
    intrinsic    = batch["intrinsic"].to(device)
    depth_map    = batch["depth_map"].to(device)
    normal_map   = batch["normal_map"].to(device)
    boxes        = batch["boxes"].to(device)
    masks        = batch["masks"].to(device)
    amodal_masks = batch["amodal_masks"].to(device)
    near_plane   = batch["near_plane"].to(device)
    far_plane    = batch["far_plane"].to(device)
    voxels       = batch["voxel_grid"]["voxels"].to(device)
    extents      = batch["voxel_grid"]["extents"].to(device)
    cam_from_obj = batch["obj_poses"]["poses"].to(device)
    scales       = batch["obj_poses"]["scales"].to(device)

    n_objects = len(boxes)
    if n_objects == 0:
        return None

    point_map = depth2cloud(depth_map, intrinsic).permute(2, 0, 1)

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_chunks = 0

    for start in range(0, n_objects, obj_chunk):
        end = min(start + obj_chunk, n_objects)
        idx = slice(start, end)

        roi_feats, grid_centers = model._compute_roi_features(
            rgb, intrinsic, point_map,
            boxes[idx], masks[idx], amodal_masks[idx], normal_map,
            near_plane, far_plane,
            perturb=True,
        )

        with torch.cuda.amp.autocast():
            logits = model_dp(roi_feats)

        gt_occ = compute_gt_occupancy(
            grid_centers, cam_from_obj[idx], scales[idx], extents[idx], voxels[idx]
        )
        pos_frac   = gt_occ.mean().clamp(min=1e-3)
        pos_weight = (1.0 - pos_frac) / pos_frac

        with torch.cuda.amp.autocast():
            chunk_loss = F.binary_cross_entropy_with_logits(
                logits, gt_occ, pos_weight=pos_weight
            )

        scaler.scale(chunk_loss).backward()
        total_loss = total_loss + chunk_loss.detach()
        n_chunks += 1

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return (total_loss / n_chunks).item()


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_step(model, model_dp, scenes, device, obj_chunk=32):
    ious = []
    for batch in scenes:
        n_objects = len(batch["boxes"])
        if n_objects == 0:
            continue

        rgb          = batch["rgb"].to(device)
        intrinsic    = batch["intrinsic"].to(device)
        depth_map    = batch["depth_map"].to(device)
        normal_map   = batch["normal_map"].to(device)
        boxes        = batch["boxes"].to(device)
        masks        = batch["masks"].to(device)
        amodal_masks = batch["amodal_masks"].to(device)
        near_plane   = batch["near_plane"].to(device)
        far_plane    = batch["far_plane"].to(device)
        voxels       = batch["voxel_grid"]["voxels"].to(device)
        extents      = batch["voxel_grid"]["extents"].to(device)
        cam_from_obj = batch["obj_poses"]["poses"].to(device)
        scales       = batch["obj_poses"]["scales"].to(device)

        point_map = depth2cloud(depth_map, intrinsic).permute(2, 0, 1)

        inter_total = torch.tensor(0.0, device=device)
        union_total = torch.tensor(0.0, device=device)

        for start in range(0, n_objects, obj_chunk):
            end = min(start + obj_chunk, n_objects)
            idx = slice(start, end)

            roi_feats, grid_centers = model._compute_roi_features(
                rgb, intrinsic, point_map,
                boxes[idx], masks[idx], amodal_masks[idx], normal_map,
                near_plane, far_plane,
            )
            with torch.cuda.amp.autocast():
                logits = model_dp(roi_feats)

            gt_occ   = compute_gt_occupancy(
                grid_centers, cam_from_obj[idx], scales[idx], extents[idx], voxels[idx]
            )
            pred_occ = logits.float().sigmoid() > 0.5

            inter_total += (pred_occ & gt_occ.bool()).float().sum()
            union_total += (pred_occ | gt_occ.bool()).float().sum()

        ious.append((inter_total / union_total.clamp(min=1e-6)).item())

    return sum(ious) / len(ious) if ious else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="/tmp/cob3d/v2")
    parser.add_argument("--checkpoint",  default="checkpoints/fcon.pt",
                        help="Path to save/load checkpoints")
    parser.add_argument("--resume",      default=None,
                        help="Path to an existing checkpoint to resume from")
    parser.add_argument("--epochs",      type=int,   default=125)
    parser.add_argument("--start_epoch", type=int,   default=0,
                        help="For old-format checkpoints (weights only): treat this as the "
                             "epoch already trained, so the cosine LR schedule resumes from "
                             "the correct position instead of restarting from 0.")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="DataLoader workers for prefetching")
    parser.add_argument("--target_scale", type=int, default=800)
    parser.add_argument("--val_only",    action="store_true")
    parser.add_argument("--val_scenes",  type=int,   default=None,
                        help="Limit validation to this many scenes (faster)")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ---- Dataset ----
    print(f"Loading dataset from {args.data_root} ...")
    dset = COB3D.load(args.data_root, target_scale=args.target_scale)

    train_ids = list(dset.scenes["train"])
    val_ids   = list(dset.scenes["val"])
    if args.val_scenes:
        val_ids = val_ids[:args.val_scenes]

    print(f"  Train: {len(train_ids)} scenes | Val: {len(val_ids)} scenes")

    # ---- Model ----
    model = FCON(n_depth_bins=96, patch_size=64).to(device)
    resume_ckpt = None

    if args.resume:
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        # Support both old format (bare state_dict) and new format (dict with keys)
        if isinstance(resume_ckpt, dict) and "model" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model"])
        else:
            model.load_state_dict(resume_ckpt)
            resume_ckpt = None  # old format — no scheduler/optimizer state to restore
            if args.start_epoch > 0:
                print(f"  Old-format checkpoint: will fast-forward scheduler to epoch {args.start_epoch}.")
    elif not args.val_only:
        print("Training from scratch.")

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        model_dp = torch.nn.DataParallel(model)
        print(f"Using DataParallel across {n_gpus} GPUs.")
    else:
        model_dp = model
        print("Single GPU / CPU — DataParallel not used.")

    # ---- DataLoaders ----
    use_persistent = args.num_workers > 0
    val_dset   = SceneDataset(dset, val_ids)
    val_loader = DataLoader(
        val_dset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=list_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
    )

    if args.val_only:
        print("\n=== Validation ===")
        model.eval()
        ious, skipped = [], 0
        for i, scenes in enumerate(val_loader):
            iou = val_step(model, model_dp, scenes, device)
            if iou is None:
                skipped += 1
            else:
                ious.append(iou)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(val_ids)}] mean IoU: {sum(ious)/len(ious):.4f}")
        print(f"\nVal IoU: {sum(ious)/len(ious):.4f}  (skipped {skipped})")
        return

    # ---- Training loop ----
    # Adam to match paper; AMP GradScaler for fp16
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    start_epoch  = 0
    best_val_iou = -1.0

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch  = resume_ckpt["epoch"] + 1
        best_val_iou = resume_ckpt.get("best_val_iou", -1.0)
        print(f"  Resuming from epoch {start_epoch}, best val IoU so far: {best_val_iou:.4f}")
    elif args.start_epoch > 0:
        # Old-format checkpoint loaded (weights only) — fast-forward the cosine scheduler
        # so LR picks up at the right point rather than restarting from lr_max.
        start_epoch = args.start_epoch
        for _ in range(args.start_epoch):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Fast-forwarded scheduler {args.start_epoch} steps → LR = {current_lr:.6f}")

    ckpt_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]  # e.g. "fcon_v2"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dset   = SceneDataset(dset, train_ids)
    train_loader = DataLoader(
        train_dset,
        batch_size=1,   # one scene per step; DataLoader handles prefetch
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, skipped = [], 0
        t0 = time.time()

        for i, scenes in enumerate(train_loader):
            loss = train_step(model, model_dp, scenes[0], device, optimizer, scaler)
            if loss is None:
                skipped += 1
            else:
                losses.append(loss)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch+1} [{i+1}/{len(train_ids)}] "
                    f"loss={sum(losses[-200:])/len(losses[-200:]):.4f}  "
                    f"elapsed={elapsed:.0f}s"
                )

        train_loss = sum(losses) / max(len(losses), 1)
        scheduler.step()

        # ---- Validation ----
        model.eval()
        ious = []
        for scenes in val_loader:
            iou = val_step(model, model_dp, scenes, device)
            if iou is not None:
                ious.append(iou)
        val_iou = sum(ious) / max(len(ious), 1)

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_iou={val_iou:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}  skipped={skipped}"
        )

        # Unwrap compiled/DataParallel model for saving
        raw_model = model_dp.module if hasattr(model_dp, "module") else model_dp
        raw_model = getattr(raw_model, "_orig_mod", raw_model)  # unwrap torch.compile

        full_ckpt = {
            "model":        raw_model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "epoch":        epoch,
            "best_val_iou": best_val_iou,
        }

        # Save latest
        torch.save(full_ckpt, args.checkpoint)
        print(f"  Checkpoint saved to {args.checkpoint}")

        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            full_ckpt["best_val_iou"] = best_val_iou
            best_path = os.path.join(ckpt_dir, f"{ckpt_stem}_best.pt")
            torch.save(full_ckpt, best_path)
            print(f"  New best val IoU {val_iou:.4f} — saved to {best_path}")

        # Snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            snap_path = os.path.join(ckpt_dir, f"{ckpt_stem}_epoch{epoch+1:04d}.pt")
            torch.save(full_ckpt, snap_path)
            print(f"  Epoch snapshot saved to {snap_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
