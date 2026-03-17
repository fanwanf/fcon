import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from frustum import Frustum
from layers.conv_blobs import ConvBlobSpec2d, ConvBlobSpec3d
from layers.conv_layer import ConvArgs3d, ConvLayer, ConvLayerSpec, LayerOrder
from layers.nonlinearity import LeakyReluArgs
from layers.normalization import GroupNormArgs
from layers.residual_block import ResidualBlock, ResidualMode
from layers.unet import UNetSimple
from torch_utils import Lambda, enlarge_box_roi, expand_dim

# ---------------------------------------------------------------------------
# Input channel layout for the 3D-UNet
# ---------------------------------------------------------------------------
#
# The input volume has shape (N, C_IN, D, H, W) where:
#
#   Foreground channels (8):  placed at the depth bin of each FOREGROUND pixel
#     [0:3]  RGB                  — object appearance
#     [3]    visible mask         — where the object is seen (always 1 here)
#     [4]    amodal mask          — full object footprint including occluded parts
#     [5:8]  surface normals      — face orientation, strongly constrains depth extent
#
#   Background channels (7):  placed at the depth bin of each BACKGROUND pixel
#     [8:11] RGB                  — background appearance (conveyor, floor, wall)
#     [11:14] surface normals     — background surface orientation (floor ≈ horizontal)
#     [14]   background indicator — constant 1.0; distinguishes measured bg from empty voxels
#
# Total: C_IN = 15
#
# Key insight for depth estimation:
#   The background channel encodes WHERE the support surface is (its depth bin)
#   and WHAT orientation it has (its normals).  For a box on a conveyor the
#   background voxels mark the belt depth, giving the model an explicit anchor
#   for the bottom of each box — information that was completely absent before.

C_FG = 8   # foreground feature channels
C_BG = 7   # background feature channels
C_IN = C_FG + C_BG   # 15 total input channels to the 3D-UNet


@attr.s(kw_only=True, eq=False, repr=False)
class FCON(nn.Module):
    n_depth_bins: int = attr.ib()
    patch_size: int = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()

        gn = lambda n: GroupNormArgs(num_per_group=n, affine=False)

        # 3D-UNet — input is now C_IN=15 channels instead of 5
        unet_channels = [32, 64, 128, 256]
        unet_spec = ConvBlobSpec3d(
            channels=unet_channels, strides=[2**i for i in range(len(unet_channels))]
        )
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=GroupNormArgs(num_groups=4),
            conv_args=ConvArgs3d(),
        )
        self.feature_module = nn.Sequential(
            ConvLayer(
                C_IN, unet_spec.channels[0], spec=attr.evolve(conv_spec, norm_args=gn(1))
            ),
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
            Lambda(lambda blobs: blobs[0]),
            ConvLayer(
                unet_spec.channels[0],
                1,
                spec=attr.evolve(conv_spec, norm_args=gn(4)).as_end_spec(),
            ),
        )

        # 2D-UNet
        unet_channels = [64, 128, 256, 512]
        unet_spec = ConvBlobSpec2d(
            channels=(self.n_depth_bins, *unet_channels),
            strides=(1, *[2**i for i in range(1, len(unet_channels) + 1)]),
        )
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=GroupNormArgs(num_groups=4),
        )
        self.unet2d = nn.Sequential(
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
            Lambda(lambda blobs: blobs[0]),
        )

        # Final predictor/decoder module
        dconv2d = 128
        self.predictor = nn.Sequential(
            ConvLayer(
                self.n_depth_bins, dconv2d, spec=attr.evolve(conv_spec, norm_args=gn(1))
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ConvLayer(
                dconv2d,
                self.n_depth_bins,
                spec=ConvLayerSpec(
                    layer_order=LayerOrder.NORM_NONLIN_CONV
                ).as_end_spec(),
            ),
        )

    def _get_frustums(
        self,
        near_plane: torch.Tensor,
        far_plane: torch.Tensor,
        point_map: torch.Tensor,
        intrinsic: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        im_size: torch.Tensor,
        perturb: bool,
    ):
        boxes = enlarge_box_roi(boxes, im_size, 1.2)
        n_objects = len(boxes)
        near_planes = (
            expand_dim(point_map[2], dim=0, shape=[n_objects])
            .masked_fill(~masks, np.inf)
            .flatten(1, 2)
            .min(1)
            .values
        )
        far_planes = torch.zeros_like(near_planes).add(far_plane)

        near_plane_offset = 0.05
        far_plane_offset = 0.0
        if perturb:
            # TODO-7: Wide depth-range randomization.
            # Forces the model to infer box depth from background/normal evidence
            # rather than reading it implicitly from the frustum range.
            # COB3D far_plane is always ~floor depth, so without this the model
            # can cheat: box_height ≈ far_plane - near_plane.
            #
            # near: push 5–35 cm above the closest visible point (uniform)
            # far:  push 0–50 cm past the floor/conveyor surface (uniform)
            near_plane_offset = torch.rand_like(near_planes).mul(0.30).add(0.05)  # [0.05, 0.35]
            far_plane_offset = -torch.rand_like(far_planes).mul(0.50)             # [-0.50, 0.0]
            # net: far_planes -= far_plane_offset  →  far_planes += [0, 0.50]

        near_planes = near_planes - near_plane_offset
        far_planes = far_planes - far_plane_offset

        return Frustum(
            roi=boxes,
            near_plane=near_planes,
            far_plane=far_planes,
            intrinsic=intrinsic,
        )

    def _voxelize_roi_point_clouds(
        self,
        frustums: Frustum,
        point_map: torch.Tensor,
        masks_p: torch.Tensor,
    ):
        """Voxelize the depth map into foreground and background volumes.

        Parameters
        ----------
        frustums : Frustum
        point_map : (3, H, W) point cloud in camera space
        masks_p : (N, 1, patch_size, patch_size) float — ROI-aligned visible mask

        Returns
        -------
        fg_voxels : (N, D, H, W) bool — depth bins occupied by FOREGROUND pixels
        bg_voxels : (N, D, H, W) bool — depth bins occupied by BACKGROUND pixels
        """
        rois = F.pad(frustums.roi, (1, 0), value=0.0)

        # ROI-align full depth map (includes both fg and bg pixels)
        depth = torchvision.ops.roi_align(
            point_map[None, None, 2],
            rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        ).squeeze(1)  # (N, patch_size, patch_size)

        eps = 1e-3
        z_normed = (depth - frustums.near_plane[:, None, None]) / (
            frustums.far_plane[:, None, None] - frustums.near_plane[:, None, None]
        )
        z_endpoints = torch.linspace(0, 1, self.n_depth_bins + 1, device=rois.device)
        z_idx = torch.bucketize(z_normed.clamp(eps, 1 - eps), z_endpoints).sub(1)

        # One-hot encode every pixel's depth bin: (N, D, H, W)
        all_voxels = F.one_hot(z_idx, self.n_depth_bins).permute(0, 3, 1, 2).bool()

        # Split by foreground/background using the ROI-aligned mask.
        # masks_p is (N, 1, H, W); broadcast across the D dimension is automatic.
        fg_pixel = masks_p.squeeze(1) > 0.5   # (N, H, W) bool
        fg_voxels = all_voxels &  fg_pixel.unsqueeze(1)  # (N, D, H, W)
        bg_voxels = all_voxels & ~fg_pixel.unsqueeze(1)  # (N, D, H, W)

        return fg_voxels, bg_voxels

    def _compute_roi_features(
        self,
        rgb: torch.Tensor,
        intrinsic: torch.Tensor,
        point_map: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        amodal_masks: torch.Tensor,
        normal_map: torch.Tensor,
        near_plane: torch.Tensor,
        far_plane: torch.Tensor,
        perturb: bool = False,
    ):
        """Non-parallelizable stage: frustum setup, ROI align, voxelization.

        Parameters
        ----------
        rgb          : (3, H, W)
        intrinsic    : (3, 3)
        point_map    : (3, H, W)  XYZ point cloud in camera space
        boxes        : (N, 4)     bounding boxes [x1,y1,x2,y2]
        masks        : (N, H, W)  bool visible segmentation masks
        amodal_masks : (N, H, W)  bool amodal (full) segmentation masks
        normal_map   : (3, H, W)  unit surface normals in camera space
        near_plane   : scalar
        far_plane    : scalar
        perturb      : bool

        Returns
        -------
        roi_voxelized_features : (N, C_IN, D, patch_size, patch_size)
        grid_pts_cam           : (N, D, patch_size, patch_size, 3)
        """
        device = rgb.device

        im_size = torch.tensor(rgb.shape[-2:], device=device)
        frustums = self._get_frustums(
            near_plane, far_plane, point_map, intrinsic,
            boxes, masks, im_size, perturb=perturb,
        )
        frustum_voxel_size = torch.tensor(
            [self.n_depth_bins, self.patch_size, self.patch_size], device=device
        )
        grid_pts_cam = frustums.to_grid(frustum_voxel_size)

        # ---- ROI-align all 2D inputs to (N, C, patch_size, patch_size) ----

        # Enlarged-box ROIs (same coordinate frame as the frustum)
        rois = F.pad(frustums.roi, (1, 0), value=0.0)  # (N, 5): [batch_idx, x1,y1,x2,y2]

        # Per-object ROIs using original (non-enlarged) boxes, one ROI per mask image
        mask_rois = torch.cat(
            [torch.arange(len(boxes), device=device)[:, None], boxes], dim=1
        )  # (N, 5)

        # RGB from enlarged box — gives background context in margins
        rgb_p = torchvision.ops.roi_align(
            rgb[None],
            rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )  # (N, 3, patch_size, patch_size)

        # Visible mask from original box — marks foreground within the patch
        masks_p = torchvision.ops.roi_align(
            masks[:, None].float(),
            mask_rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )  # (N, 1, patch_size, patch_size)

        # Amodal mask from original box — full object footprint including occluded regions
        amodal_p = torchvision.ops.roi_align(
            amodal_masks[:, None].float(),
            mask_rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )  # (N, 1, patch_size, patch_size)

        # Surface normals from enlarged box (bilinear), then re-normalize per pixel
        normals_p = F.normalize(
            torchvision.ops.roi_align(
                normal_map[None],
                rois,
                output_size=(self.patch_size, self.patch_size),
                sampling_ratio=2,
            ),
            dim=1,  # normalize along the channel (3,) dimension at each spatial location
        )  # (N, 3, patch_size, patch_size)

        # ---- Voxelization: split depth into foreground and background volumes ----

        fg_voxels, bg_voxels = self._voxelize_roi_point_clouds(
            frustums, point_map, masks_p
        )  # each (N, D, patch_size, patch_size) bool

        # ---- Build per-voxel feature volumes via einsum ----
        # einsum "bdhw,bchw->bcdhw":  for each (h,w) pixel, scatter its C-dim
        # feature vector into depth bin d (all other bins stay zero).

        # Foreground volume: 8 channels
        #   RGB(3) + visible_mask(1) + amodal_mask(1) + normals(3)
        fg_feats = torch.cat(
            [rgb_p, masks_p, amodal_p, normals_p], dim=1
        )  # (N, 8, H, W)
        fg_volume = torch.einsum(
            "bdhw,bchw->bcdhw", fg_voxels.float(), fg_feats
        )  # (N, 8, D, H, W)

        # Background volume: 7 channels
        #   RGB(3) + normals(3) + bg_indicator(1)
        # bg_indicator is 1.0 for every pixel; after the einsum it will be 1.0 exactly
        # at the measured background depth bin and 0 everywhere else.  This lets the
        # model distinguish "measured background surface" from "empty/unobserved voxel".
        bg_indicator = torch.ones(
            len(boxes), 1, self.patch_size, self.patch_size, device=device
        )
        bg_feats = torch.cat(
            [rgb_p, normals_p, bg_indicator], dim=1
        )  # (N, 7, H, W)
        bg_volume = torch.einsum(
            "bdhw,bchw->bcdhw", bg_voxels.float(), bg_feats
        )  # (N, 7, D, H, W)

        # Concatenate along channel dim → (N, 15, D, H, W)
        roi_voxelized_features = torch.cat([fg_volume, bg_volume], dim=1)

        return roi_voxelized_features, grid_pts_cam

    def forward(self, roi_voxelized_features: torch.Tensor) -> torch.Tensor:
        """Parallelizable NN stage: 3D-UNet + 2D-UNet + predictor."""
        feats = self.feature_module(roi_voxelized_features).squeeze(1)
        feats = self.unet2d(feats)
        return self.predictor(feats)

    def predict(
        self,
        rgb: torch.Tensor,
        intrinsic: torch.Tensor,
        point_map: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        amodal_masks: torch.Tensor,
        normal_map: torch.Tensor,
        near_plane: torch.Tensor,
        far_plane: torch.Tensor,
        perturb: bool = False,
    ) -> dict[str, torch.Tensor]:
        roi_voxelized_features, grid_pts_cam = self._compute_roi_features(
            rgb, intrinsic, point_map, boxes, masks,
            amodal_masks, normal_map,
            near_plane, far_plane, perturb=perturb,
        )
        logits = self(roi_voxelized_features)
        return {
            "logits": logits,
            "grid_centers": grid_pts_cam,
        }
