import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _valid_mask(self, disp_gt):
        """
        disp_gt: [B,H,W]
        """
        valid = (disp_gt > 0) & (disp_gt < self.max_disp)
        return valid

    def _disp_to_3d(self, disp, rays_L, cam_L):
        """
        disp: [B,1,H,W]
        returns: [B,H,W,3] world coordinates
        """
        B = disp.shape[0]

        fx = cam_L["K"][:, 0, 0].view(B, 1, 1, 1)
        baseline = cam_L["baseline"].view(B, 1, 1, 1)

        depth = (fx * baseline) / (disp + 1e-6)

        C_L = -torch.bmm(
            cam_L["R"].transpose(1, 2),
            cam_L["t"].unsqueeze(-1)
        ).squeeze(-1)  # [B,3]

        X = (
            C_L.view(B, 1, 1, 3) +
            depth.permute(0, 2, 3, 1) *
            rays_L.permute(0, 2, 3, 1)
        )
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

        disp_gt = input_data["disp"].to(device)          # [B,H,W]
        disp_pred = model_pred["disp_pred"]              # [B,1,H,W]

        B, _, H, W = disp_pred.shape

        # -------------------------------
        # Valid mask
        # -------------------------------
        valid = self._valid_mask(disp_gt)
        valid = valid.unsqueeze(1)                        # [B,1,H,W]

        # -------------------------------
        # Disparity loss (stabilizer)
        # -------------------------------
        disp_loss = torch.tensor(0.0, device=device)

        if "init_disp" in model_pred:
            disp_init = model_pred["init_disp"]
            disp_init = F.interpolate(
                disp_init, scale_factor=4,
                mode="bilinear", align_corners=True
            ) * 4

            disp_loss = F.smooth_l1_loss(
                disp_init[valid],
                disp_gt.unsqueeze(1)[valid],
                reduction="mean"
            )

        # -------------------------------
        # Camera geometry
        # -------------------------------
        camera = input_data["camera"]

        cam_L = {
            "K": camera["K_L"].to(device),
            "R": camera["R_L"].to(device),
            "t": camera["t_L"].to(device),
            "baseline": camera["baseline"].to(device)
        }
        cam_R = {
            "K": camera["K_R"].to(device),
            "R": camera["R_R"].to(device),
            "t": camera["t_R"].to(device)
        }

        rays_L, _ = compute_rays_batch(
            H, W,
            cam_L["K"], cam_L["R"], cam_L["t"]
        )

        pixel_grid = make_pixel_grid(B, H, W, device)

        # -------------------------------
        # Reprojection loss (MAIN)
        # -------------------------------
        offset_pred, _ = project_left_to_right_offset(
            disp_pred, rays_L, cam_L, cam_R, pixel_grid
        )

        offset_gt, _ = project_left_to_right_offset(
            disp_gt.unsqueeze(1), rays_L, cam_L, cam_R, pixel_grid
        )

        reproj_loss = (
            offset_pred - offset_gt
        ).abs()

        reproj_loss = reproj_loss[valid.expand_as(reproj_loss)].mean()

        # -------------------------------
        # Optional 3D loss
        # -------------------------------
        loss_3d = torch.tensor(0.0, device=device)

        if self.use_3d_loss:
            X_pred = self._disp_to_3d(disp_pred, rays_L, cam_L)
            X_gt = self._disp_to_3d(
                disp_gt.unsqueeze(1), rays_L, cam_L
            )

            loss_3d = (X_pred - X_gt).abs()
            loss_3d = loss_3d[valid.permute(0,2,3,1).expand_as(loss_3d)].mean()

        # -------------------------------
        # Total loss
        # -------------------------------
        total_loss = (
            self.lambda_disp * disp_loss +
            self.lambda_reproj * reproj_loss +
            self.lambda_3d * loss_3d
        )

        loss_info = {
            "scalar/loss_disp": disp_loss.item(),
            "scalar/loss_reproj": reproj_loss.item(),
            "scalar/loss_3d": loss_3d.item(),
            "scalar/loss_total": total_loss.item(),
        }

        return total_loss, loss_info
