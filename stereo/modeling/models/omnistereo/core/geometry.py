# core/geometry.py
from math import isinf
import os
import sys
import torch
import torch.nn.functional as F
import logging
from core.utils.utils import bilinear_sampler

logger = logging.getLogger("geometry")

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")


# ============================================================
# 原有：Rectified Stereo Geo Encoding Volume（不动）
# ============================================================

class Combined_Geo_Encoding_Volume:
    def __init__(
        self,
        init_fmap1,
        init_fmap2,
        geo_volume,
        num_levels=2,
        dx=None,
        use_camera_geometry: bool = False,
    ):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx
        self.use_camera_geometry = use_camera_geometry

        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)

        for _ in range(self.num_levels - 1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, low_memory=False):
        """
        disp: (B,1,H,W)
        coords:
        - rectified stereo: absolute x-coords (u)
        - camera-conditioned: offset (dx, dy)
        """
        b, _, h, w = disp.shape
        out_pyramid = []

        if self.dx is None:
            raise ValueError("Combined_Geo_Encoding_Volume requires dx (corr_radius offsets). Got dx=None.")
        # expected shape: (1, 1, 2r+1, 1)
        dx = self.dx.to(disp.device).to(disp.dtype)
        # build (1, 1, 2r+1, 2):(delta_x, delta_y=0)
        delta = torch.cat([dx, torch.zeros_like(dx)], dim=-1)

        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]

            if self.use_camera_geometry:
                # coords is OFFSET (u_R-u_L, v_R-v_L) from reprojection: (B*H*W, 1, 1, 2)
                # Map to a pseudo-disparity for 1D cost volume sampling:
                # rectified stereo: u_R = u_L - disp => offset_x = -disp => disp ~= -offset_x
                offset_x = coords[..., 0:1]  # (B*H*W, 1, 1, 1)
                disp_cam = -offset_x
                x0 = dx + disp_cam / (2 ** i)
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
            else:
                x0 = dx + disp.reshape(b*h*w,1,1,1) / (2**i)
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)

            geo_volume = bilinear_sampler(
                geo_volume,
                coords_lvl,
                low_memory=low_memory
            )
            geo_volume = geo_volume.reshape(b, h, w, -1)

            # ---------- init_corr ----------
            init_corr = self.init_corr_pyramid[i]
            if self.use_camera_geometry:
                # Sample around projected right x-coordinate u_R = u_L + offset_x
                offset_x = coords[..., 0:1]  # (B*H*W, 1, 1, 1)
                xL = (
                    torch.arange(w, device=disp.device, dtype=disp.dtype)
                    .view(1, 1, w)
                    .repeat(b, h, 1)
                    .reshape(b*h*w, 1, 1, 1)
                )
                uR = xL + offset_x
                init_x0 = uR / (2 ** i) + dx
                y0 = torch.zeros_like(init_x0)
                init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            else:
                init_x0 = (
                    coords.reshape(b*h*w,1,1,1) / (2**i)
                    - disp.reshape(b*h*w,1,1,1) / (2**i)
                    + dx
                )
                y0 = torch.zeros_like(init_x0)
                init_coords_lvl = torch.cat([init_x0, y0], dim=-1)

            init_corr = bilinear_sampler(
                init_corr,
                init_coords_lvl,
                low_memory=low_memory
            )
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        with torch.cuda.amp.autocast(enabled=False):
            corr = torch.einsum(
                "aijk,aijh->ajkh",
                F.normalize(fmap1.float(), dim=1),
                F.normalize(fmap2.float(), dim=1)
            )
        return corr.reshape(B, H, W1, 1, W2)


# ============================================================
# 新增：Camera-Conditioned Geometry（核心）
# ============================================================

def make_pixel_grid(B, H, W, device):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([xs, ys], dim=-1).float()  # (H,W,2)
    return grid.unsqueeze(0).repeat(B,1,1,1)     # (B,H,W,2)


def compute_rays_batch(H, W, K, R, t):
    """
    Return rays in world frame and camera centers.
    K,R,t are world->camera.
    """
    K = K.float()
    R = R.float()
    t = t.float()

    B = K.shape[0]
    device = K.device

    pix = make_pixel_grid(B, H, W, device)
    ones = torch.ones_like(pix[..., :1])
    pix_h = torch.cat([pix, ones], dim=-1).reshape(B, H*W, 3)

    Kinv = torch.inverse(K)
    rays_cam = torch.bmm(pix_h, Kinv.transpose(1,2))
    rays_world = torch.bmm(rays_cam, R.transpose(1,2))
    rays_world = rays_world / (torch.norm(rays_world, dim=-1, keepdim=True) + 1e-8)
    rays_world = rays_world.view(B,H,W,3).permute(0,3,1,2)
    cam_center = -torch.bmm(R.transpose(1,2), t.unsqueeze(-1)).squeeze(-1)
    return rays_world, cam_center


def project_left_to_right_offset(
    disp, rays_L, cam_L, cam_R, pixel_grid_L, eps=1e-6
):
    """
    Core operator:
    (u_L,v_L) + disparity → depth → 3D → project → (u_R,v_R)
    return OFFSET = (u_R-u_L, v_R-v_L) for bilinear_sampler
    """
    rays_L = rays_L.float()
    pixel_grid_L = pixel_grid_L.float()
    K_L = cam_L["K"].float()
    R_L = cam_L["R"].float()
    t_L = cam_L["t"].float()
    K_R = cam_R["K"].float()
    R_R = cam_R["R"].float()
    t_R = cam_R["t"].float()
    baseline = cam_L["baseline"].float()

    B, _, H, W = disp.shape
    device = disp.device

    fx = K_L[:, 0, 0]

    if torch.isnan(disp).sum().item() > 0 or torch.isinf(disp).sum().item() > 0:
        logger.info(f"[geometry][project_left_to_right_offset] disp is nan number :{torch.isnan(disp).sum().item()}"
                    f" disp is inf number :{torch.isinf(disp).sum().item()}")
    """
    logging if disp is valid (z_min=1e-3, max_depth=1e4, disp_min=0.1)
    """

    # ---- robustness: clamp disp/depth to avoid exploding projections ----
    disp_safe = torch.clamp(disp, min=1e-3)
    depth = (fx.view(B,1,1,1) * baseline.view(B,1,1,1)) / (disp_safe + eps)
    depth = torch.clamp(depth, min=0.0, max=1e3)
    
    if torch.isnan(depth).any().item() > 0 or torch.isinf(depth).any().item() > 0:
        logger.info(f"[geometry][project_left_to_right_offset] depth is nan number :{torch.isnan(depth).sum().item()}"
                    f" depth is inf number :{torch.isinf(depth).sum().item()}")

    C_L = -torch.bmm(
        R_L.transpose(1,2),
        t_L.unsqueeze(-1)
    ).squeeze(-1)

    rays_world = rays_L.permute(0, 2, 3, 1).reshape(B, -1, 3)
    
    rays_cam = torch.bmm(R_L, rays_world.transpose(1, 2)).transpose(1, 2)
    ray_z = rays_cam[..., 2].view(B, H, W, 1)
    
    depth_z = depth.permute(0, 2, 3, 1)
    ray_z = torch.where(ray_z.abs() < eps, ray_z.sign() * eps, ray_z)
    t_scale = depth_z / (ray_z + eps)

    X = C_L.view(B, 1, 1, 3) + t_scale * rays_L.permute(0, 2, 3, 1)

    Xr = torch.bmm(
        R_R,
        X.reshape(B,-1,3).transpose(1,2)
    ).transpose(1,2) + t_R.view(B,1,3)

    proj = torch.bmm(
        K_R,
        Xr.transpose(1,2)
    ).transpose(1,2)

    z = proj[..., 2]
    z = torch.where(z.abs() < eps, z.sign() * eps, z)
    u = proj[...,0] / (z + eps)
    v = proj[...,1] / (z + eps)
    coords_R = torch.stack([u,v], dim=-1).view(B,H,W,2)

    offset = coords_R - pixel_grid_L
    offset = torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)
    return offset, depth