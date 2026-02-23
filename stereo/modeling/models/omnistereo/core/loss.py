import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger =logging.getLogger("loss")
from core.geometry import (
    compute_rays_batch,
    make_pixel_grid,
    project_left_to_right_offset
)


class GeometryStereoLoss(nn.Module):
    """
    Geometry-aware stereo loss.

    Components:
      - Disparity loss (stability)
      - Reprojection loss (main geometry loss)
      - Optional 3D point loss
    """

    def __init__(
        self,
        max_disp,
        lambda_disp=1.0,
        lambda_reproj=0.5,
        lambda_3d=0.0,
        use_3d_loss=False
    ):
        super().__init__()
        self.max_disp = max_disp

        self.lambda_disp = lambda_disp
        self.lambda_reproj = lambda_reproj
        self.lambda_3d = lambda_3d

        self.use_3d_loss = use_3d_loss

    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------

    def _valid_mask(self, disp):
        """
        disp_gt: [B,H,W]
        """
        valid = (disp > 0) & (disp < self.max_disp) &(torch.isfinite(disp))
        return valid

    def _disp_to_3d(self, disp, rays_L, cam_L):
        """
        disp: [B,1,H,W]
        returns: [B,H,W,3] world coordinates
        """
        B = disp.shape[0]

        fx = cam_L["K"][:, 0, 0].view(B, 1, 1, 1)
        baseline = cam_L["baseline"].view(B, 1, 1, 1)

        # Add numerical stability check
        disp_safe = torch.clamp(disp, min=1e-6, max=self.max_disp)
        depth = (fx * baseline) / disp_safe

        C_L = -torch.bmm(
            cam_L["R"].transpose(1, 2),
            cam_L["t"].unsqueeze(-1)
        ).squeeze(-1).view(B, 1, 1, 3)  # [B,3]

        X = C_L + depth.permute(0, 2, 3, 1) * rays_L.permute(0, 2, 3, 1)
        return X

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------

    def forward(self, model_pred, input_data):
        """
        model_pred: dict from FoundationStereo
        input_data: batch dict
        """

        device = model_pred["disp_pred"].device

        disp_gt = input_data["disp"].to(device,non_blocking=True).unsqueeze(1)         # [B,H,W]
        disp_pred = model_pred["disp_pred"]              # [B,1,H,W]

        B, _, H, W = disp_pred.shape

        camera = {k:v.to(device,non_blocking=True) for k,v in input_data["camera"].items()}

        # -------------------------------
        # Valid mask
        # -------------------------------
    
        valid_disp_gt_mask = self._valid_mask(disp_gt)

        # -------------------------------
        # Disparity loss (stabilizer)
        # -------------------------------
        # Initialize with zero from computation graph to maintain gradient connection
        loss_disp = torch.tensor(0.0, device=device)

        if "init_disp" in model_pred:
            disp_init = F.interpolate(
                model_pred["init_disp"], scale_factor=4,
                mode="bilinear", align_corners=True
            ) * 4

            loss_disp = F.smooth_l1_loss(
                    disp_init[valid_disp_gt_mask],
                    disp_gt[valid_disp_gt_mask],
                    reduction="mean"
            )
        # -------------------------------
        # Camera geometry
        # -------------------------------

        cam_L = {
            "K": camera["K_L"],
            "R": camera["R_L"],
            "t": camera["t_L"],
            "baseline": camera["baseline"]
        }
        cam_R = {
            "K": camera["K_R"],
            "R": camera["R_R"],
            "t": camera["t_R"]
        }

        rays_L, _ = compute_rays_batch(
            H, W,
            cam_L["K"], cam_L["R"], cam_L["t"]
        )

        pixel_grid = make_pixel_grid(B, H, W, device)

        # -------------------------------
        # Reprojection loss (MAIN)
        # -------------------------------
        # Get valid_mask from projection function (consistent with omnistereo forward)
        offset_pred, _, valid_mask_pred = project_left_to_right_offset(
            disp_pred, rays_L, cam_L, cam_R, pixel_grid
        )

        offset_gt, _, valid_mask_gt = project_left_to_right_offset(
            disp_gt, rays_L, cam_L, cam_R, pixel_grid
        )
          
        mask_valid_reproj = valid_mask_pred & valid_mask_gt & valid_disp_gt_mask.squeeze(1) # [B,H,W]
        
        # Log how many pixels were filtered out (save original count before modifying valid_reproj)
        n_original = valid_mask_gt.sum().item()
        n_reproj_original = mask_valid_reproj.sum().item()
        if logger.isEnabledFor(logging.INFO) and n_original > 0:
            ratio = n_reproj_original / n_original
            logger.info(
                f"({ratio*100:.1f}%) valid projections ratio")
        
        # Apply stronger mask to reprojection loss
        mask_valid_reproj = mask_valid_reproj.unsqueeze(-1)  # [B,H,W,1]
        
        diff = (offset_pred - offset_gt).float()

        reproj_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=1.0)  # [B,H,W,2]

        # Extract valid reprojection loss values
        valid_reproj_expanded = mask_valid_reproj.expand_as(reproj_loss)  # [B,H,W,2]
        reproj_loss = reproj_loss[valid_reproj_expanded]
        loss_reproj = reproj_loss.mean() if reproj_loss.any() else (diff.sum() * 0)
        
        # -------------------------------
        # Optional 3D loss
        # -------------------------------
        # Initialize with zero from computation graph to maintain gradient connection
        loss_3d = torch.tensor(0.0, device=device)

        if self.use_3d_loss:
            X_pred = self._disp_to_3d(disp_pred, rays_L, cam_L)
            X_gt = self._disp_to_3d(
                disp_gt, rays_L, cam_L
            )
            
            loss_3d_map = torch.abs(X_pred - X_gt)
            mask_3d = valid_disp_gt_mask.permute(0, 2, 3, 1).expand_as(loss_3d_map)
            loss_3d = loss_3d_map[mask_3d].mean() if mask_3d.any() else (X_pred.sum() * 0)
        # -------------------------------
        # Total loss
        # -------------------------------
        total_loss = (
            self.lambda_disp * loss_disp +
            self.lambda_reproj * loss_reproj +
            self.lambda_3d * loss_3d
        )

        loss_info = {
            "scalar/loss_disp": loss_disp.item(),
            "scalar/loss_reproj": loss_reproj.item(),
            "scalar/loss_3d": loss_3d.item(),
            "scalar/loss_total": total_loss.item(),
        }

        return total_loss, loss_info