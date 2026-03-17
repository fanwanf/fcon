"""
Evaluate the pretrained F-CON model on COB-3D-v2 val set.
Reproduces the metrics from Table I of the paper:
  - CD-L1, CD-L2  (Chamfer distance, point clouds scaled to GT bbox longest edge = 10)
  - F1 at tau = 0.1, 0.3, 0.5
  - Box-IoU, Box-IoG, Box-F1

GPU-accelerated via pytorch3d knn_points; multi-GPU via torch.multiprocessing.

Usage:
    python evaluate.py --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt
    python evaluate.py --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt --n_workers 8
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.ops import knn_points
from scipy.ndimage import label as scipy_label

from cob3d_dataset import COB3D
from fcon_model import FCON
from torch_utils import depth2cloud, interp3d, to_np, to_torch, transform_points


# ---------------------------------------------------------------------------
# Post-processing: largest connected component
# ---------------------------------------------------------------------------

def largest_connected_component(occ_vol: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary 3D volume.

    Removes floating fragments that would produce spurious predicted surfaces
    and inflate bounding-box estimates.
    """
    labeled, n = scipy_label(occ_vol)
    if n <= 1:
        return occ_vol
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background label (label 0 = empty)
    return (labeled == sizes.argmax()).astype(occ_vol.dtype)


def apply_lcc(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binarize a (N, D, H, W) probability volume, apply LCC per object, return float tensor."""
    binary = (probs > threshold).cpu().numpy()
    for i in range(binary.shape[0]):
        binary[i] = largest_connected_component(binary[i])
    return torch.from_numpy(binary.astype(np.float32)).to(probs.device)


# ---------------------------------------------------------------------------
# GPU-accelerated point-cloud metrics
# ---------------------------------------------------------------------------

def chamfer_gpu(x: torch.Tensor, y: torch.Tensor):
    """Chamfer-L1 and Chamfer-L2. x, y: (N, 3) float32 tensors on the same device."""
    xb = x.unsqueeze(0)   # (1, N, 3)
    yb = y.unsqueeze(0)   # (1, M, 3)
    knn_xy = knn_points(xb, yb, K=1, return_sorted=False)
    knn_yx = knn_points(yb, xb, K=1, return_sorted=False)
    d2_xy = knn_xy.dists[0, :, 0]   # (N,) squared distances
    d2_yx = knn_yx.dists[0, :, 0]   # (M,)
    cd_l2 = (d2_xy.mean() + d2_yx.mean()).item() / 2.0
    cd_l1 = (d2_xy.sqrt().mean() + d2_yx.sqrt().mean()).item() / 2.0
    return cd_l1, cd_l2


def f1_score_gpu(x: torch.Tensor, y: torch.Tensor, tau: float):
    """F1 at threshold tau. x, y: (N, 3) float32 tensors."""
    xb = x.unsqueeze(0)
    yb = y.unsqueeze(0)
    knn_xy = knn_points(xb, yb, K=1, return_sorted=False)
    knn_yx = knn_points(yb, xb, K=1, return_sorted=False)
    d_xy = knn_xy.dists[0, :, 0].sqrt()
    d_yx = knn_yx.dists[0, :, 0].sqrt()
    precision = (d_xy < tau).float().mean().item()
    recall    = (d_yx < tau).float().mean().item()
    if precision + recall < 1e-9:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def scale_to_gt_bbox_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    """Scale so the longest edge of the GT bbox = 10."""
    gt_min = gt_pts.min(0).values
    gt_max = gt_pts.max(0).values
    longest_edge = (gt_max - gt_min).max().item()
    if longest_edge < 1e-9:
        return pred_pts, gt_pts
    scale = 10.0 / longest_edge
    return pred_pts * scale, gt_pts * scale


# ---------------------------------------------------------------------------
# 3-D bounding-box metrics (CPU, cheap after scaling)
# ---------------------------------------------------------------------------

def min_volume_bbox(pts: np.ndarray, n_rotations: int = 64):
    """Minimum-volume bounding box via random rotations."""
    best_vol, best_dims, best_center = np.inf, None, None
    for _ in range(n_rotations):
        q = np.random.randn(4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        pts_r = pts @ R.T
        lo, hi = pts_r.min(0), pts_r.max(0)
        dims = hi - lo; vol = dims.prod()
        if vol < best_vol:
            best_vol = vol; best_dims = dims
            best_center = R.T @ ((lo + hi) / 2)
    return best_center, best_dims


def box_iou_iog_f1(pred_pts: np.ndarray, gt_pts: np.ndarray, n_rotations: int = 64):
    """Box-IoU, Box-IoG, Box-F1 via minimum-volume bounding boxes."""
    pred_center, pred_dims = min_volume_bbox(pred_pts, n_rotations)
    gt_center,   gt_dims   = min_volume_bbox(gt_pts,   n_rotations)

    p_lo = pred_center - pred_dims / 2; p_hi = pred_center + pred_dims / 2
    g_lo = gt_center   - gt_dims   / 2; g_hi = gt_center   + gt_dims   / 2

    inter_dims = np.maximum(np.minimum(p_hi, g_hi) - np.maximum(p_lo, g_lo), 0)
    inter_vol  = inter_dims.prod()
    pred_vol = pred_dims.prod(); gt_vol = gt_dims.prod()
    union_vol = pred_vol + gt_vol - inter_vol

    iou = inter_vol / (union_vol + 1e-9)
    iog = inter_vol / (gt_vol    + 1e-9)
    f1  = 2 * iou * iog / (iou + iog + 1e-9)
    return float(iou), float(iog), float(f1)


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_scene(model, scene_path, mesh_dir, device, n_pts=16384, obj_chunk=8):
    f = np.load(scene_path, mmap_mode="r", allow_pickle=True)
    mesh_ids = list(f["mesh_ids"])

    segm = f["segm"].item()
    vf   = to_np(segm["masks"]).sum((1, 2)) / to_np(segm["amodal_masks"]).sum((1, 2))
    keep = vf > 0.1

    mesh_ids     = [mesh_ids[i] for i, k in enumerate(keep) if k]
    cam_from_obj = torch.from_numpy(to_np(f["obj_poses"].item()["poses"])[keep]).float()
    scales       = torch.from_numpy(to_np(f["obj_poses"].item()["scales"])[keep]).float()
    extents      = torch.from_numpy(to_np(f["voxel_grid"].item()["extents"])[keep]).float()

    rgb          = torch.from_numpy(to_np(f["rgb"])).float().to(device)
    intrinsic    = torch.from_numpy(to_np(f["intrinsic"])).float().to(device)
    depth_map    = torch.from_numpy(to_np(f["depth_map"])).float().to(device)
    normal_map   = torch.from_numpy(to_np(f["normal_map"])).float().to(device)
    near_plane   = torch.tensor(float(f["near_plane"])).to(device)
    far_plane    = torch.tensor(float(f["far_plane"])).to(device)
    boxes        = torch.from_numpy(to_np(segm["boxes"])[keep]).float().to(device)
    masks        = torch.from_numpy(to_np(segm["masks"])[keep]).to(device)
    amodal_masks = torch.from_numpy(to_np(segm["amodal_masks"])[keep]).to(device)

    # Rescale to target_scale=800 (same as training)
    h, w = rgb.shape[-2:]
    scale_factor = 800.0 / np.sqrt(h * w)
    rgb          = F.interpolate(rgb[None], scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]
    intrinsic    = intrinsic.clone(); intrinsic[0:2] *= scale_factor
    depth_map    = F.interpolate(depth_map[None, None], scale_factor=scale_factor, mode="nearest")[0, 0]
    # Normals: bilinear resize then re-normalize per pixel to restore unit length
    normal_map   = F.normalize(
        F.interpolate(normal_map[None], scale_factor=scale_factor, mode="bilinear", align_corners=False)[0],
        dim=0,
    )
    boxes        = boxes * scale_factor
    masks        = F.interpolate(masks[:, None].byte(), scale_factor=scale_factor, mode="nearest")[:, 0].bool()
    amodal_masks = F.interpolate(amodal_masks[:, None].byte(), scale_factor=scale_factor, mode="nearest")[:, 0].bool()

    point_map = depth2cloud(depth_map, intrinsic).permute(2, 0, 1)

    n_objects = len(mesh_ids)
    if n_objects == 0:
        return []

    results = []
    for start in range(0, n_objects, obj_chunk):
        end = min(start + obj_chunk, n_objects)
        idx = slice(start, end)

        out = model.predict(rgb, intrinsic, point_map,
                            boxes[idx], masks[idx], amodal_masks[idx], normal_map,
                            near_plane, far_plane)
        logits       = out["logits"].float()
        grid_centers = out["grid_centers"]

        probs = logits.sigmoid()
        # LCC filter: drop floating fragments before marching cubes (TODO-9)
        probs_filtered = apply_lcc(probs, threshold=0.5)
        verts_idx_lst, faces_lst = marching_cubes(probs_filtered, 0.5, return_local_coords=False)

        cam_from_obj_b = cam_from_obj[idx].to(device)
        scales_b       = scales[idx].to(device)
        extents_b      = extents[idx].to(device)

        for i in range(end - start):
            mid = mesh_ids[start + i]
            mesh_path = os.path.join(mesh_dir, f"{mid}.stl")
            if not os.path.exists(mesh_path):
                continue

            # GT point cloud
            try:
                gt_mesh = trimesh.load(mesh_path, force="mesh")
                gt_pts_obj = trimesh.sample.sample_surface(gt_mesh, n_pts)[0].astype(np.float32)
            except Exception:
                continue

            # STL meshes are already in physical object space; transform directly to camera
            gt_pts_cam = (to_np(cam_from_obj_b[i, :3, :3]) @ gt_pts_obj.T).T \
                         + to_np(cam_from_obj_b[i, :3, 3])

            # Predicted point cloud via marching cubes
            verts_idx = verts_idx_lst[i]
            faces     = faces_lst[i]
            if len(verts_idx) == 0 or len(faces) == 0:
                continue

            verts_cam = to_np(interp3d(
                grid_centers[i].permute(3, 0, 1, 2),
                verts_idx.flip(-1),
            ))  # (V, 3) in camera space

            try:
                mesh = trimesh.Trimesh(vertices=verts_cam, faces=to_np(faces))
                pred_pts_cam = trimesh.sample.sample_surface(mesh, n_pts)[0].astype(np.float32)
            except Exception:
                continue

            if len(pred_pts_cam) == 0:
                continue

            # Move both to GPU for fast metric computation
            pred_t = torch.from_numpy(pred_pts_cam).float().to(device)
            gt_t   = torch.from_numpy(gt_pts_cam).float().to(device)
            pred_s, gt_s = scale_to_gt_bbox_gpu(pred_t, gt_t)

            cd_l1, cd_l2 = chamfer_gpu(pred_s, gt_s)
            f1_01 = f1_score_gpu(pred_s, gt_s, 0.1)
            f1_03 = f1_score_gpu(pred_s, gt_s, 0.3)
            f1_05 = f1_score_gpu(pred_s, gt_s, 0.5)

            # Box metrics (CPU, cheap)
            pred_np = pred_s.cpu().numpy()
            gt_np   = gt_s.cpu().numpy()
            box_iou, box_iog, box_f1 = box_iou_iog_f1(pred_np, gt_np)

            results.append(dict(
                cd_l1=cd_l1, cd_l2=cd_l2,
                f1_01=f1_01, f1_03=f1_03, f1_05=f1_05,
                box_iou=box_iou, box_iog=box_iog, box_f1=box_f1,
            ))
    return results


# ---------------------------------------------------------------------------
# Worker for multi-GPU evaluation
# ---------------------------------------------------------------------------

def worker_fn(rank, args, scene_ids, result_queue):
    """Each worker runs on GPU `rank` and evaluates a subset of scenes."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = FCON(n_depth_bins=96, patch_size=64).to(device)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    mesh_dir = os.path.join(args.data_root, "meshes")
    local_results = []
    t0 = time.time()

    for i, sid in enumerate(scene_ids):
        scene_path = os.path.join(args.data_root, "scenes", f"{sid}.npz")
        try:
            res = evaluate_scene(model, scene_path, mesh_dir, device,
                                 n_pts=args.n_pts)
            local_results.extend(res)
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  GPU{rank}: [{i+1}/{len(scene_ids)}] {len(local_results)} objects | {elapsed:.0f}s",
                  flush=True)

    result_queue.put(local_results)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(all_results, n_scenes):
    n = len(all_results)
    if n == 0:
        print("No valid objects found.")
        return
    cd_l1 = np.mean([r["cd_l1"]   for r in all_results])
    cd_l2 = np.mean([r["cd_l2"]   for r in all_results])
    f1_01 = np.mean([r["f1_01"]   for r in all_results])
    f1_03 = np.mean([r["f1_03"]   for r in all_results])
    f1_05 = np.mean([r["f1_05"]   for r in all_results])
    b_iou = np.mean([r["box_iou"] for r in all_results])
    b_iog = np.mean([r["box_iog"] for r in all_results])
    b_f1  = np.mean([r["box_f1"]  for r in all_results])

    print()
    print("=" * 65)
    print("FINAL RESULTS  (F-CON pretrained, COB-3D-v2 val)")
    print("=" * 65)
    print(f"Scenes: {n_scenes}  |  Objects evaluated: {n}")
    print()
    print("Shape completion (GT bbox longest edge = 10):")
    print(f"  CD-L1  (down)  : {cd_l1:.4f}   [paper: 0.4229]")
    print(f"  CD-L2  (down)  : {cd_l2:.4f}   [paper: 0.1157]")
    print(f"  F1@0.1 (up)    : {f1_01:.4f}   [paper: 0.4664]")
    print(f"  F1@0.3 (up)    : {f1_03:.4f}   [paper: 0.8928]")
    print(f"  F1@0.5 (up)    : {f1_05:.4f}   [paper: 0.9600]")
    print()
    print("3-D bounding-box metrics:")
    print(f"  Box-IoU (up)   : {b_iou:.4f}   [paper: 0.6809]")
    print(f"  Box-IoG (up)   : {b_iog:.4f}   [paper: 0.7686]")
    print(f"  Box-F1  (up)   : {b_f1:.4f}   [paper: 0.7485]")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="/tmp/cob3d/v2")
    parser.add_argument("--checkpoint", default="checkpoints/fcon.pt")
    parser.add_argument("--n_scenes",   type=int, default=None,
                        help="Limit to first N val scenes (default: all 696)")
    parser.add_argument("--n_pts",      type=int, default=16384)
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Number of GPUs to use in parallel (default 1)")
    args = parser.parse_args()

    dset = COB3D.load(args.data_root, target_scale=800)
    val_ids = list(dset.scenes["val"])
    if args.n_scenes:
        val_ids = val_ids[:args.n_scenes]

    n_gpus = min(args.n_workers, torch.cuda.device_count())
    print(f"GPUs available : {torch.cuda.device_count()}")
    print(f"Workers        : {n_gpus}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Scenes         : {len(val_ids)}")
    print()

    t0 = time.time()

    if n_gpus <= 1:
        # Single-GPU path
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model = FCON(n_depth_bins=96, patch_size=64).to(device)
        ckpt  = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)\n")

        mesh_dir    = os.path.join(args.data_root, "meshes")
        all_results = []
        for i, sid in enumerate(val_ids):
            scene_path = os.path.join(args.data_root, "scenes", f"{sid}.npz")
            try:
                res = evaluate_scene(model, scene_path, mesh_dir, device, n_pts=args.n_pts)
                all_results.extend(res)
            except Exception:
                pass

            if (i + 1) % 50 == 0 or (i + 1) == len(val_ids):
                n = len(all_results)
                elapsed = time.time() - t0
                if n > 0:
                    cd_l1 = np.mean([r["cd_l1"] for r in all_results])
                    f1_03 = np.mean([r["f1_03"] for r in all_results])
                    print(f"[{i+1:3d}/{len(val_ids)}] {n} objects | "
                          f"CD-L1={cd_l1:.4f} F1@0.3={f1_03:.4f} | {elapsed:.0f}s",
                          flush=True)

        print_summary(all_results, len(val_ids))

    else:
        # Multi-GPU path via torch.multiprocessing
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # Split scenes across workers
        chunks = [val_ids[i::n_gpus] for i in range(n_gpus)]
        result_queue = mp.Queue()
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(target=worker_fn,
                           args=(rank, args, chunks[rank], result_queue))
            p.start()
            processes.append(p)

        all_results = []
        for _ in range(n_gpus):
            all_results.extend(result_queue.get())

        for p in processes:
            p.join()

        elapsed = time.time() - t0
        print(f"\nAll workers done in {elapsed:.0f}s")
        print_summary(all_results, len(val_ids))


if __name__ == "__main__":
    main()
