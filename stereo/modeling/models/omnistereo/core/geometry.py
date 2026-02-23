import torch,pdb,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *
import logging

logger = logging.getLogger("geometry")

# Helper function to get current process rank
class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume,pixel_grid_L = None, num_levels=2, dx=None,use_camera_geometry: bool = False):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx
        self.use_camera_geometry = use_camera_geometry
        self.pixel_grid_L = pixel_grid_L
        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d).contiguous()

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
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

        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]

            if self.use_camera_geometry:
                # coords is already (dx, dy)
                offset_x = coords[..., 0:1]  # (B*H*W, 1, 1, 1)
                disp_cam = -offset_x
                x0 = dx + disp_cam / (2 ** i)
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
            else:
                x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / (2**i)
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)

            geo_volume = bilinear_sampler(
                geo_volume,
                coords_lvl,
                low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]

            if self.use_camera_geometry:
                u_L = self.pixel_grid_L[..., 0:1].reshape(b*h*w, 1, 1, 1)/(2**i)
                offset_x = coords[..., 0:1]  # (B*H*W, 1, 1, 1)
                u_R = u_L + offset_x / (2**i)
                x0 = dx + u_R
                y0 = torch.zeros_like(x0)
                init_coords_lvl = torch.cat([x0, y0], dim=-1) 
            else:
                init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + self.dx   # X on right image
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
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape(B, D, H, W1)
        fmap2 = fmap2.reshape(B, D, H, W2)
        with torch.cuda.amp.autocast(enabled=False):
          corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        corr = corr.reshape(B, H, W1, 1, W2)
        return corr

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
    rays_world = torch.bmm(rays_cam, R)
    # 考虑不进行normalize ->Z深度等同于相机坐标系中的深度
    # rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)
    
    rays_world = rays_world.view(B,H,W,3).permute(0,3,1,2)
    cam_center = -torch.bmm(R.transpose(1,2), t.unsqueeze(-1)).squeeze(-1)
    return rays_world, cam_center


def project_left_to_right_offset(
    disp, rays_L, cam_L, cam_R, pixel_grid_L, eps=1e-6, max_depth=500,disp_threshold=0.1
):
    """
    Core operator:
    (u_L,v_L) + disparity → depth → 3D → project → (u_R,v_R)
    return OFFSET = (u_R-u_L, v_R-v_L) for bilinear_sampler
    """
    B, _, H, W = disp.shape
    device = disp.device
    with torch.cuda.amp.autocast(enabled=False):
        fx = cam_L["K"][:, 0, 0].float().view(B, 1, 1, 1)
        baseline = cam_L["baseline"].float().view(B, 1, 1, 1)

        min_disp = (fx * baseline) / (max_depth + float(eps))

        valid_mask = (disp.float() > min_disp) & torch.isfinite(disp.float())
        disp_valid = torch.clamp(valid_mask, min=min_disp)

        depth = (fx * baseline) / (disp_valid + float(eps))
        depth = torch.clamp(depth, min=0.0, max=float(max_depth))

        C_L = -torch.bmm(
            cam_L["R"].float().transpose(1, 2),
            cam_L["t"].float().unsqueeze(-1)
        ).squeeze(-1)

        X = C_L.view(B, 1, 1, 3) + depth.permute(0, 2, 3, 1) * rays_L.float().permute(0, 2, 3, 1)

        R_r = cam_R["R"].float()
        t_r = cam_R["t"].float()
        if t_r.dim() == 2: t_r = t_r.unsqueeze(-1)

        Xr = torch.bmm(R_r,X.view(B, H*W, 3).float().transpose(1, 2)) + t_r

        proj = torch.bmm(
            cam_R["K"].float(),
            Xr
        )

        proj_x = proj[:, 0:1, :]
        proj_y = proj[:, 1:2, :]
        proj_z = proj[:, 2:3, :]

        valid_z_mask = proj_z > eps
        proj_z = torch.where(valid_z_mask, proj_z, torch.ones_like(proj_z))

        u = proj_x / proj_z
        v = proj_y / proj_z
    
        coords_R = torch.cat([u, v], dim=-1).view(B, H, W, 2)  # shape: (B, H, W, 2)
        in_box_u = (coords_R[...,0] >=0) &(coords_R[...,0] < W)
        in_box_v = (coords_R[...,1] >=0) &(coords_R[...,1] < H)
        in_box = in_box_u & in_box_v

        offset = coords_R - pixel_grid_L.float()
        # 有效掩码 (disp + z + offset + proj right in bounding box)
        valid_mask = valid_mask.squeeze(1) & valid_z_mask.view(B,H,W) & in_box.view(B,H,W) \
        & torch.isfinite(offset[..., 0]) & torch.isfinite(offset[..., 1])
        offset = torch.where(valid_mask.unsqueeze(-1), offset, torch.zeros_like(offset))

        offset = offset.to(disp.dtype)
        depth = depth.to(disp.dtype)

        return offset, depth, valid_mask # (B,H,W,2), (B,1,H,W), (B,H,W)
    