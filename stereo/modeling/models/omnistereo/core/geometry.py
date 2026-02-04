# core/geometry.py
import os
import sys
import torch
import torch.nn.functional as F
import logging
from core.utils.utils import bilinear_sampler

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

logger =logging.getLogger('geometry')
# ============================================================
# 原有：Rectified Stereo Geo Encoding Volume（不动）
# ============================================================

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx

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
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []

        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = self.dx + disp.reshape(b*h*w,1,1,1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)

            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b,h,w,-1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w,1,1,1)/2**i - disp.reshape(b*h*w,1,1,1)/2**i + self.dx
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b,h,w,-1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0,3,1,2).contiguous()

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
    B = K.shape[0]
    device = K.device

    pix = make_pixel_grid(B, H, W, device)
    ones = torch.ones_like(pix[..., :1])
    logger.info(f'[geometry] [compute_rays_batch] pix.shape: {pix.shape}')
    pix_h = torch.cat([pix, ones], dim=-1).reshape(B, H*W, 3)

    Kinv = torch.inverse(K)
    rays_cam = torch.bmm(pix_h, Kinv.transpose(1,2))# 像素坐标系 -> 相机坐标系
    rays_world = torch.bmm(rays_cam, R.transpose(1,2))# 相机坐标系 -> 世界坐标系
    rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)# 归一化(代表射线方向)

    rays_world = rays_world.view(B,H,W,3).permute(0,3,1,2) # 射线的方向
    cam_center = -torch.bmm(R.transpose(1,2), t.unsqueeze(-1)).squeeze(-1) # 相机中心
    return rays_world, cam_center


def project_left_to_right_offset(
    disp, rays_L, cam_L, cam_R, pixel_grid_L, eps=1e-6
):
    """
    Core operator:
    (u_L,v_L) + disparity → depth → 3D → project → (u_R,v_R)
    return OFFSET = (u_R-u_L, v_R-v_L) for bilinear_sampler
    """
    B, _, H, W = disp.shape
    device = disp.device

    fx = cam_L["K"][:,0,0]
    baseline = cam_L["baseline"]

    depth = (fx.view(B,1,1,1) * baseline.view(B,1,1,1)) / (disp + eps)

    C_L = -torch.bmm(
        cam_L["R"].transpose(1,2),
        cam_L["t"].unsqueeze(-1)
    ).squeeze(-1)

    X = C_L.view(B,1,1,3) + depth.permute(0,2,3,1) * rays_L.permute(0,2,3,1)

    Xr = torch.bmm(
        cam_R["R"],
        X.reshape(B,-1,3).transpose(1,2)
    ).transpose(1,2) + cam_R["t"].view(B,1,3)

    proj = torch.bmm(
        cam_R["K"],
        Xr.transpose(1,2)
    ).transpose(1,2)

    u = proj[...,0] / (proj[...,2] + eps)
    v = proj[...,1] / (proj[...,2] + eps)
    coords_R = torch.stack([u,v], dim=-1).view(B,H,W,2)

    offset = coords_R - pixel_grid_L
    return offset, depth
