"""
Three-way comparison visualization on the F-CON val set, uploaded to W&B.

For each object in each scene the script uploads:
  - RGB scene image
  - GT point cloud (green) vs background point cloud (gray)
  - New model prediction (red) vs background point cloud (gray)
  - Old / author checkpoint prediction (blue) vs background point cloud (gray)
  - Per-object metrics for both models
  - Aggregated summary metrics for both models

Models compared
---------------
  new  : fcon_model.py :: FCON  (15-ch input: normals + background depth, TODO-1 & 2)
  old  : fcon_model_v1.py :: FCON_V1  (5-ch input, original paper architecture)

Usage
-----
    python visualize_wandb.py \\
        --data_root  /home/fanwanf/lab42/fcon/data/v2 \\
        --ckpt_new   checkpoints/fcon_v2_best.pt \\
        --ckpt_old   checkpoints/fcon.pt \\
        --n_scenes   50 \\
        --wandb_project fcon-viz
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import wandb
from pytorch3d.ops import knn_points
from pytorch3d.ops.marching_cubes import marching_cubes

from cob3d_dataset import COB3D
from fcon_model import FCON
from fcon_model_v1 import FCON_V1
from torch_utils import depth2cloud, interp3d, to_np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chamfer_gpu(x, y):
    d2_xy = knn_points(x.unsqueeze(0), y.unsqueeze(0), K=1, return_sorted=False).dists[0, :, 0]
    d2_yx = knn_points(y.unsqueeze(0), x.unsqueeze(0), K=1, return_sorted=False).dists[0, :, 0]
    return (d2_xy.sqrt().mean() + d2_yx.sqrt().mean()).item() / 2.0, \
           (d2_xy.mean() + d2_yx.mean()).item() / 2.0


def f1_score_gpu(x, y, tau):
    d_xy = knn_points(x.unsqueeze(0), y.unsqueeze(0), K=1, return_sorted=False).dists[0, :, 0].sqrt()
    d_yx = knn_points(y.unsqueeze(0), x.unsqueeze(0), K=1, return_sorted=False).dists[0, :, 0].sqrt()
    p = (d_xy < tau).float().mean().item()
    r = (d_yx < tau).float().mean().item()
    return 2 * p * r / (p + r + 1e-9)


def scale_to_gt(pred, gt):
    longest = (gt.max(0).values - gt.min(0).values).max().item()
    if longest < 1e-9:
        return pred, gt
    s = 10.0 / longest
    return pred * s, gt * s


def min_volume_bbox(pts, n=64):
    best_vol, best_dims, best_center = np.inf, None, None
    for _ in range(n):
        q = np.random.randn(4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
        pts_r = pts @ R.T; lo, hi = pts_r.min(0), pts_r.max(0)
        dims = hi - lo; vol = dims.prod()
        if vol < best_vol:
            best_vol = vol; best_dims = dims
            best_center = R.T @ ((lo + hi) / 2)
    return best_center, best_dims


def box_metrics(p, g):
    pc, pd = min_volume_bbox(p); gc, gd = min_volume_bbox(g)
    inter = np.maximum(np.minimum(pc+pd/2, gc+gd/2) - np.maximum(pc-pd/2, gc-gd/2), 0).prod()
    iou = inter / (pd.prod() + gd.prod() - inter + 1e-9)
    iog = inter / (gd.prod() + 1e-9)
    return float(iou), float(iog), float(2*iou*iog/(iou+iog+1e-9))


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def pred_cloud_from_logits(logits_i, grid_centers_i, faces_i, verts_idx_i, n_pts):
    """Sample a point cloud from marching-cubes output for object i."""
    if len(verts_idx_i) == 0 or len(faces_i) == 0:
        return None
    verts_cam = to_np(interp3d(
        grid_centers_i.permute(3, 0, 1, 2),
        verts_idx_i.flip(-1),
    ))
    try:
        mesh = trimesh.Trimesh(vertices=verts_cam, faces=to_np(faces_i))
        pts = trimesh.sample.sample_surface(mesh, n_pts)[0].astype(np.float32)
        return pts if len(pts) > 0 else None
    except Exception:
        return None


def background_cloud(point_map, mask_hw, near_z, far_z, n_pts=1024):
    """Return (N,3) array of background scene points near this object's depth range.

    point_map : (3, H, W) float32 on CPU
    mask_hw   : (H, W) bool — True = this object's foreground
    near_z    : float
    far_z     : float
    """
    depth = point_map[2]  # (H, W)
    valid = (~mask_hw) & (depth > 0.01) & (depth >= near_z - 0.3) & (depth <= far_z + 0.1)
    pts = point_map[:, valid].T.numpy()  # (N, 3)
    if len(pts) == 0:
        return None
    if len(pts) > n_pts:
        pts = pts[np.random.choice(len(pts), n_pts, replace=False)]
    return pts


def colored_cloud(pts, rgb_uint8):
    """pts: (N,3), rgb_uint8: 3-tuple or (3,) — returns (N,6) float array."""
    c = np.broadcast_to(np.array(rgb_uint8, dtype=np.float32), (len(pts), 3))
    return np.hstack([pts, c])


# ---------------------------------------------------------------------------
# Run marching cubes on the logits of a batch
# ---------------------------------------------------------------------------

def run_mc(logits):
    probs = logits.sigmoid()
    return marching_cubes(probs, 0.5, return_local_coords=False)


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_scene(model_new, model_old, scene_path, mesh_dir, device,
                   n_pts=4096, obj_chunk=8, max_vis=3):
    scene_id = os.path.splitext(os.path.basename(scene_path))[0]
    f = np.load(scene_path, mmap_mode="r", allow_pickle=True)
    mesh_ids = list(f["mesh_ids"])

    segm = f["segm"].item()
    vf   = to_np(segm["masks"]).sum((1, 2)) / (to_np(segm["amodal_masks"]).sum((1, 2)) + 1e-9)
    keep = vf > 0.1

    mesh_ids     = [mesh_ids[i] for i, k in enumerate(keep) if k]
    cam_from_obj = torch.from_numpy(to_np(f["obj_poses"].item()["poses"])[keep]).float()

    rgb_np       = to_np(f["rgb"])                          # (3,H,W) for W&B image
    rgb          = torch.from_numpy(rgb_np).float().to(device)
    intrinsic    = torch.from_numpy(to_np(f["intrinsic"])).float().to(device)
    depth_map    = torch.from_numpy(to_np(f["depth_map"])).float().to(device)
    normal_map   = torch.from_numpy(to_np(f["normal_map"])).float().to(device)
    near_plane   = torch.tensor(float(f["near_plane"])).to(device)
    far_plane    = torch.tensor(float(f["far_plane"])).to(device)
    boxes        = torch.from_numpy(to_np(segm["boxes"])[keep]).float().to(device)
    masks        = torch.from_numpy(to_np(segm["masks"])[keep]).to(device)
    amodal_masks = torch.from_numpy(to_np(segm["amodal_masks"])[keep]).to(device)

    # Rescale to target_scale=800
    scale = 800.0 / np.sqrt(rgb.shape[-2] * rgb.shape[-1])
    rgb          = F.interpolate(rgb[None], scale_factor=scale, mode="bilinear",    align_corners=False)[0]
    intrinsic    = intrinsic.clone(); intrinsic[0:2] *= scale
    depth_map    = F.interpolate(depth_map[None, None], scale_factor=scale, mode="nearest")[0, 0]
    normal_map   = F.normalize(
        F.interpolate(normal_map[None], scale_factor=scale, mode="bilinear", align_corners=False)[0], dim=0)
    boxes        = boxes * scale
    masks        = F.interpolate(masks[:, None].byte(), scale_factor=scale, mode="nearest")[:, 0].bool()
    amodal_masks = F.interpolate(amodal_masks[:, None].byte(), scale_factor=scale, mode="nearest")[:, 0].bool()

    point_map      = depth2cloud(depth_map, intrinsic).permute(2, 0, 1)  # (3,H,W) on device
    point_map_cpu  = point_map.cpu()                                       # keep a CPU copy for bg cloud

    # RGB image for W&B  (H,W,3 uint8)
    rgb_img = (rgb_np.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)

    n_objects = len(mesh_ids)
    if n_objects == 0:
        return [], []

    metric_rows = []
    vis_rows    = []
    vis_count   = 0

    for start in range(0, n_objects, obj_chunk):
        end = min(start + obj_chunk, n_objects)
        idx = slice(start, end)

        # ----- New model inference -----
        out_new = model_new.predict(
            rgb, intrinsic, point_map,
            boxes[idx], masks[idx], amodal_masks[idx], normal_map,
            near_plane, far_plane,
        )
        verts_new, faces_new = run_mc(out_new["logits"].float())

        # ----- Old model inference -----
        out_old = model_old.predict(
            rgb, intrinsic, point_map,
            boxes[idx], masks[idx],
            near_plane, far_plane,
        )
        verts_old, faces_old = run_mc(out_old["logits"].float())

        cam_from_obj_b = cam_from_obj[idx].to(device)

        for i in range(end - start):
            global_idx = start + i
            mid = mesh_ids[global_idx]
            mesh_path = os.path.join(mesh_dir, f"{mid}.stl")
            if not os.path.exists(mesh_path):
                continue

            # GT point cloud
            try:
                gt_mesh = trimesh.load(mesh_path, force="mesh")
                gt_pts_obj = trimesh.sample.sample_surface(gt_mesh, n_pts)[0].astype(np.float32)
            except Exception:
                continue
            gt_pts_cam = (to_np(cam_from_obj_b[i, :3, :3]) @ gt_pts_obj.T).T \
                         + to_np(cam_from_obj_b[i, :3, 3])

            # Predicted clouds
            pred_new = pred_cloud_from_logits(
                out_new["logits"][i], out_new["grid_centers"][i],
                faces_new[i], verts_new[i], n_pts,
            )
            pred_old = pred_cloud_from_logits(
                out_old["logits"][i], out_old["grid_centers"][i],
                faces_old[i], verts_old[i], n_pts,
            )

            if pred_new is None and pred_old is None:
                continue

            # --- Compute per-object near_z for background extraction ---
            mask_hw = masks[global_idx]  # (H,W) bool on device
            depth_masked = point_map[2].masked_fill(~mask_hw, float("inf"))
            near_z = (depth_masked.min().item() - 0.05)
            far_z  = far_plane.item()

            # Background point cloud (CPU, subsampled)
            bg_pts = background_cloud(
                point_map_cpu, mask_hw.cpu(), near_z, far_z, n_pts=1024
            )

            # --- Metrics for new model ---
            def compute_metrics(pred_pts, gt_pts_cam, device):
                pred_t = torch.from_numpy(pred_pts).float().to(device)
                gt_t   = torch.from_numpy(gt_pts_cam).float().to(device)
                ps, gs = scale_to_gt(pred_t, gt_t)
                cd_l1, cd_l2 = chamfer_gpu(ps, gs)
                return dict(
                    cd_l1=cd_l1, cd_l2=cd_l2,
                    f1_01=f1_score_gpu(ps, gs, 0.1),
                    f1_03=f1_score_gpu(ps, gs, 0.3),
                    f1_05=f1_score_gpu(ps, gs, 0.5),
                    **dict(zip(
                        ("box_iou", "box_iog", "box_f1"),
                        box_metrics(ps.cpu().numpy(), gs.cpu().numpy()),
                    )),
                )

            m_new = compute_metrics(pred_new, gt_pts_cam, device) if pred_new is not None else None
            m_old = compute_metrics(pred_old, gt_pts_cam, device) if pred_old is not None else None

            metric_rows.append(dict(
                scene_id=scene_id, obj_idx=global_idx, mesh_id=mid,
                **{f"new_{k}": v for k, v in (m_new or {}).items()},
                **{f"old_{k}": v for k, v in (m_old or {}).items()},
            ))

            # --- Visualization ---
            if vis_count < max_vis:
                # Scale all clouds to GT bbox = 10 (same reference frame for visual comparison)
                gt_t = torch.from_numpy(gt_pts_cam).float().to(device)
                longest = (gt_t.max(0).values - gt_t.min(0).values).max().item()
                s = (10.0 / longest) if longest > 1e-9 else 1.0

                gt_scaled  = gt_pts_cam * s
                bg_scaled  = (bg_pts * s) if bg_pts is not None else None
                new_scaled = (pred_new * s) if pred_new is not None else None
                old_scaled = (pred_old * s) if pred_old is not None else None

                def make_cloud(*pairs):
                    """pairs = [(pts, rgb_tuple), ...]  → stacked (N,6)"""
                    parts = [colored_cloud(p, c) for p, c in pairs if p is not None]
                    return np.vstack(parts) if parts else None

                # Colors: GT=green, new=red, old=blue, bg=light gray
                GT_C  = (50,  200, 50)
                NEW_C = (220, 50,  50)
                OLD_C = (50,  100, 220)
                BG_C  = (180, 180, 180)

                gt_vs_bg  = make_cloud((gt_scaled, GT_C),  (bg_scaled, BG_C))
                new_vs_bg = make_cloud((new_scaled, NEW_C), (bg_scaled, BG_C))
                old_vs_bg = make_cloud((old_scaled, OLD_C), (bg_scaled, BG_C))

                vis_rows.append(dict(
                    scene_id=scene_id,
                    obj_idx=global_idx,
                    mesh_id=mid,
                    rgb_img=rgb_img,
                    gt_vs_bg=gt_vs_bg,
                    new_vs_bg=new_vs_bg,
                    old_vs_bg=old_vs_bg,
                    new_cd_l1=m_new["cd_l1"] if m_new else float("nan"),
                    old_cd_l1=m_old["cd_l1"] if m_old else float("nan"),
                    new_f1_03=m_new["f1_03"] if m_new else float("nan"),
                    old_f1_03=m_old["f1_03"] if m_old else float("nan"),
                ))
                vis_count += 1

    return metric_rows, vis_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="/home/fanwanf/lab42/fcon/data/v2")
    parser.add_argument("--ckpt_new",      default="checkpoints/fcon_v2_best.pt",
                        help="New model checkpoint (FCON with normals+bg)")
    parser.add_argument("--ckpt_old",      default="checkpoints/fcon.pt",
                        help="Author's original checkpoint (FCON_V1)")
    parser.add_argument("--n_scenes",      type=int, default=50)
    parser.add_argument("--n_pts",         type=int, default=4096)
    parser.add_argument("--max_vis_per_scene", type=int, default=2)
    parser.add_argument("--wandb_project", default="fcon-viz")
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--wandb_name",    default=None)
    parser.add_argument("--device",        default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Load new model ----
    print(f"Loading new model: {args.ckpt_new}")
    ckpt_new = torch.load(args.ckpt_new, map_location="cpu")
    if isinstance(ckpt_new, dict) and "model" in ckpt_new:
        epoch_new = ckpt_new.get("epoch")
        best_iou  = ckpt_new.get("best_val_iou")
        sd_new    = ckpt_new["model"]
        print(f"  epoch={epoch_new}, best_val_iou={best_iou:.4f}")
    else:
        epoch_new, best_iou, sd_new = None, None, ckpt_new
    model_new = FCON(n_depth_bins=96, patch_size=64).to(device)
    model_new.load_state_dict(sd_new)
    model_new.eval()
    print(f"  {sum(p.numel() for p in model_new.parameters())/1e6:.1f}M params")

    # ---- Load old model ----
    print(f"Loading old model: {args.ckpt_old}")
    ckpt_old = torch.load(args.ckpt_old, map_location="cpu")
    sd_old   = ckpt_old["model"] if isinstance(ckpt_old, dict) and "model" in ckpt_old else ckpt_old
    model_old = FCON_V1(n_depth_bins=96, patch_size=64).to(device)
    model_old.load_state_dict(sd_old)
    model_old.eval()
    print(f"  {sum(p.numel() for p in model_old.parameters())/1e6:.1f}M params")

    # ---- Dataset ----
    print(f"Loading dataset from {args.data_root} ...")
    dset    = COB3D.load(args.data_root, target_scale=800)
    val_ids = list(dset.scenes["val"])[:args.n_scenes]
    mesh_dir = os.path.join(args.data_root, "meshes")
    print(f"  Evaluating {len(val_ids)} scenes\n")

    # ---- W&B init ----
    run_name = args.wandb_name or f"compare_ep{epoch_new or 'unk'}_n{len(val_ids)}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=dict(
            ckpt_new=args.ckpt_new, epoch_new=epoch_new, best_val_iou=best_iou,
            ckpt_old=args.ckpt_old,
            n_scenes=len(val_ids), n_pts=args.n_pts,
        ),
    )
    print(f"W&B: {run.url}\n")

    # Metrics table
    metric_cols = [
        "scene_id", "obj_idx", "mesh_id",
        "new_cd_l1", "new_cd_l2", "new_f1_01", "new_f1_03", "new_f1_05",
        "new_box_iou", "new_box_iog", "new_box_f1",
        "old_cd_l1", "old_cd_l2", "old_f1_01", "old_f1_03", "old_f1_05",
        "old_box_iou", "old_box_iog", "old_box_f1",
    ]
    metrics_table = wandb.Table(columns=metric_cols)

    # Visualization table
    vis_cols = [
        "scene_id", "obj_idx", "mesh_id",
        "rgb_image",
        "GT_vs_background",
        "new_model_vs_background",
        "old_model_vs_background",
        "new_cd_l1", "old_cd_l1",
        "new_f1@0.3", "old_f1@0.3",
    ]
    vis_table = wandb.Table(columns=vis_cols)

    all_metrics = []
    t0 = time.time()

    for i, sid in enumerate(val_ids):
        scene_path = os.path.join(args.data_root, "scenes", f"{sid}.npz")
        try:
            m_rows, v_rows = evaluate_scene(
                model_new, model_old, scene_path, mesh_dir, device,
                n_pts=args.n_pts, max_vis=args.max_vis_per_scene,
            )
        except Exception as e:
            print(f"  Scene {sid}: ERROR — {e}")
            continue

        all_metrics.extend(m_rows)

        for r in m_rows:
            metrics_table.add_data(
                r.get("scene_id"), r.get("obj_idx"), r.get("mesh_id"),
                r.get("new_cd_l1"), r.get("new_cd_l2"),
                r.get("new_f1_01"), r.get("new_f1_03"), r.get("new_f1_05"),
                r.get("new_box_iou"), r.get("new_box_iog"), r.get("new_box_f1"),
                r.get("old_cd_l1"), r.get("old_cd_l2"),
                r.get("old_f1_01"), r.get("old_f1_03"), r.get("old_f1_05"),
                r.get("old_box_iou"), r.get("old_box_iog"), r.get("old_box_f1"),
            )

        for v in v_rows:
            def obj3d(pts):
                return wandb.Object3D(pts) if pts is not None else None
            vis_table.add_data(
                v["scene_id"], v["obj_idx"], v["mesh_id"],
                wandb.Image(v["rgb_img"]),
                obj3d(v["gt_vs_bg"]),
                obj3d(v["new_vs_bg"]),
                obj3d(v["old_vs_bg"]),
                v["new_cd_l1"], v["old_cd_l1"],
                v["new_f1_03"], v["old_f1_03"],
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(val_ids):
            n = len(all_metrics)
            elapsed = time.time() - t0
            if n > 0:
                def mean_key(k):
                    vals = [r[k] for r in all_metrics if r.get(k) is not None]
                    return np.mean(vals) if vals else float("nan")
                print(
                    f"[{i+1:3d}/{len(val_ids)}] {n} objects | "
                    f"new CD-L1={mean_key('new_cd_l1'):.4f} F1@0.3={mean_key('new_f1_03'):.4f} | "
                    f"old CD-L1={mean_key('old_cd_l1'):.4f} F1@0.3={mean_key('old_f1_03'):.4f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

    # ---- Summary ----
    n = len(all_metrics)
    if n == 0:
        print("No valid objects found.")
        run.finish()
        return

    def agg(key):
        vals = [r[key] for r in all_metrics if r.get(key) is not None]
        return np.mean(vals) if vals else float("nan")

    summary = {}
    for prefix in ("new", "old"):
        for m in ("cd_l1", "cd_l2", "f1_01", "f1_03", "f1_05", "box_iou", "box_iog", "box_f1"):
            summary[f"{prefix}/{m}"] = agg(f"{prefix}_{m}")
    summary["n_objects"] = n

    run.log({"metrics/per_object": metrics_table})
    run.log({"viz/comparison": vis_table})
    run.summary.update(summary)

    print()
    print("=" * 72)
    print(f"RESULTS  ({n} objects, {len(val_ids)} scenes)")
    print(f"{'Metric':<12}  {'New model':>12}  {'Old (paper)':>12}  {'Paper target':>12}")
    print("-" * 72)
    rows = [
        ("CD-L1",   "new/cd_l1",   "old/cd_l1",   0.4229, "down"),
        ("CD-L2",   "new/cd_l2",   "old/cd_l2",   0.1157, "down"),
        ("F1@0.1",  "new/f1_01",   "old/f1_01",   0.4664, "up"),
        ("F1@0.3",  "new/f1_03",   "old/f1_03",   0.8928, "up"),
        ("F1@0.5",  "new/f1_05",   "old/f1_05",   0.9600, "up"),
        ("Box-IoU", "new/box_iou", "old/box_iou", 0.6809, "up"),
        ("Box-IoG", "new/box_iog", "old/box_iog", 0.7686, "up"),
        ("Box-F1",  "new/box_f1",  "old/box_f1",  0.7485, "up"),
    ]
    for name, nk, ok, paper, _ in rows:
        print(f"{name:<12}  {summary[nk]:>12.4f}  {summary[ok]:>12.4f}  {paper:>12.4f}")
    print("=" * 72)
    print(f"\nW&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
