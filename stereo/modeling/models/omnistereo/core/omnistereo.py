# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import torchvision
from core.update import BasicSelectiveMultiUpdateBlock
from core.extractor import ContextNetDino, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.geometry import (
    compute_rays_batch,
    make_pixel_grid,
    project_left_to_right_offset
)
from core.submodule import (
    BasicConv, Conv3dNormActReduced, CostVolumeDisparityAttention, FeatureAtt,
    SpatialAttentionExtractor, ChannelAttentionEnhancement, BasicConv_IN, Conv2x,
    ResnetBasicBlock3D, build_gwc_volume, build_concat_volume, disparity_regression,
    context_upsample
)
from core.loss import GeometryStereoLoss
from core.utils.utils import InputPadder
# from Utils import *
import time,huggingface_hub

logger = logging.getLogger("omnistereo")

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class OmniStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28
        self.max_disp = args.max_disp

        # parameter modify according to the training of the model part
        self.lambda_disp=getattr(self.args,'lambda_disp', 1.0)
        self.lambda_reproj=getattr(self.args,'lambda_reproj', 0.5)
        self.lambda_3d=getattr(self.args,'lambda_3d', 0.1)
        
        self.disp_threshold = getattr(args, 'disp_threshold', 0.1)

        self.cnet = ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature(args)
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        # -------------------------------------------------
        # Ray encoder (pixel-level geometry injection)
        # -------------------------------------------------
        self.ray_encoder = nn.Conv2d(
            self.feature.d_out[0] + 3,
            self.feature.d_out[0],
            kernel_size=1,
            padding=0,
            bias=False
        )

        # -------------------------------------------------
        # Baseline encoder (scale conditioning, left-only)
        # -------------------------------------------------
        self.baseline_encoder = nn.Sequential(
            nn.Conv2d(1, self.feature.d_out[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feature.d_out[0]),
        )
        
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )


        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
        )


        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx

        self.criterion = GeometryStereoLoss(
            max_disp=self.max_disp,
            lambda_disp=self.lambda_disp,
            lambda_reproj=self.lambda_reproj,
            lambda_3d=self.lambda_3d,
            use_3d_loss=True
        )

        # -- frozen to training the baseline_encoder and ray_encoder --
        freeze_backbone = getattr(args, 'freeze_backbone', False) or getattr(args, 'FREEZE_BACKBONE', False)
        if freeze_backbone:
            for name, param in self.named_parameters():
                if 'ray_encoder' not in name and 'baseline_encoder' not in name:
                    param.requires_grad = False
            logger.info("Freezing backbone parameters, only training ray_encoder and baseline_encoder")


    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()


    def forward(self, data):        
        """ Estimate disparity between pair of frames """
        init_disp = None
        test_mode = not self.training
        iters = self.args.valid_iters if test_mode else self.args.train_iters
        low_memory = self.args.get('low_memory', False)

        image1 = data["left"]
        image2 = data["right"]
        if self.args.use_camera_geometry:
            camera = data["camera"]
            cam_L = {
                "K": camera["K_L"].to(image1.device),
                "R": camera["R_L"].to(image1.device),
                "t": camera["t_L"].to(image1.device),
                "baseline": camera.get("baseline", None)
            }

            cam_R = {
                "K": camera["K_R"].to(image1.device),
                "R": camera["R_R"].to(image1.device),
                "t": camera["t_R"].to(image1.device)
            } # "t" shape :(1, 3)

        B = len(image1)
        # image1 = normalize_image(image1)
        # image2 = normalize_image(image2)
        with autocast(enabled=self.args.mixed_precision):
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = self.stem_2(image1)

            if self.args.use_camera_geometry:
                _, _, h, w = features_left[0].shape
                H_img, W_img = image1.shape[-2:]
                
                # ====
                #  对相机内参K基于像素的尺寸执行缩放 
                # ====
                sx = float(w) / float(W_img)
                sy = float(h) / float(H_img)

                def scale_camera_K(K):
                    Ks = K.clone()
                    Ks[:, 0, :] *= sx
                    Ks[:, 1, :] *= sy
                    return Ks

                cam_L_sc = {**cam_L, "K": scale_camera_K(cam_L["K"])}
                cam_R_sc = {**cam_R, "K": scale_camera_K(cam_R["K"])}

                rays_L, _ = compute_rays_batch(h, w, cam_L_sc["K"], cam_L_sc["R"], cam_L_sc["t"])
                rays_R, _ = compute_rays_batch(h, w, cam_R_sc["K"], cam_R_sc["R"], cam_R_sc["t"])
                pixel_grid = make_pixel_grid(B, h, w, image1.device)

                rays_L_norm = rays_L / torch.norm(rays_L, dim=1, keepdim=True)
                rays_R_norm = rays_R / torch.norm(rays_R, dim=1, keepdim=True)
                
                baseline = cam_L["baseline"]
                baseline = baseline.view(B,1,1,1).expand(-1,1,h,w)
                baseline_feat = self.baseline_encoder(baseline.detach())  #baseline_encoder shape: (1, 224, 80, 184)

                feat_L = features_left[0]
                feat_R = features_right[0]

                featL_geo = self.ray_encoder(
                    torch.cat([feat_L, rays_L_norm], dim=1)
                ) # ray_encoder shape :(1, 224, 80, 184)
                featL_geo = featL_geo + baseline_feat
                featR_geo = self.ray_encoder(
                    torch.cat([feat_R, rays_R_norm], dim=1)
                )

                # updated feature include (geometry | direction | baseline feautre )
                features_left[0]  = featL_geo
                features_right[0] = featR_geo

            # Group-wise correlation volume (B, N_group, max_disp, H, W)
            gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4) # concatination volume 
            del left_tmp, right_tmp
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left) # cost volume aggregation

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  #(B, max_disp, H, W)
            if init_disp is None:
              init_disp = disparity_regression(prob, self.args.max_disp//4)  # Weighted  sum of disparity

            cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]   # Hidden information
            inp_list = [torch.relu(x[1]) for x in cnet_list]   # Context information list of pyramid levels
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        geo_fn = Combined_Geo_Encoding_Volume(
            features_left[0].float(),
            features_right[0].float(),
            comb_volume.float(),
            num_levels=self.args.corr_levels,
            pixel_grid_L = pixel_grid,
            dx=self.dx,
            use_camera_geometry=bool(self.args.use_camera_geometry),
        )

        b, c, h, w = features_left[0].shape
        if not self.args.use_camera_geometry:
            coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        disp_preds = []

        # GRUs iterations to update disparity (1/4 resolution)
        valid_mask = None
        for itr in range(iters):
            disp = disp.detach()
            if self.args.use_camera_geometry:
                offset, depth, valid_mask = project_left_to_right_offset(
                    disp,
                    rays_L,
                    cam_L_sc,
                    cam_R_sc,
                    pixel_grid,
                    disp_threshold=self.disp_threshold
                )
                coords_offset = offset.reshape(b*h*w, 1, 1, 2)
                geo_feat = geo_fn(disp, coords_offset, low_memory=low_memory)
            else:
                geo_feat = geo_fn(disp, coords, low_memory=low_memory)

            if self.args.use_camera_geometry and itr == 0 and not test_mode:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[omnistereo] [forward] mean dx:{offset[..., 0].mean().item()},"
                    f"mean |dy|:{offset[..., 1].abs().mean().item()}")
            with autocast(enabled=self.args.mixed_precision):
              net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)

            disp = disp + delta_disp.float()
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp.float(), mask_feat_4, stem_2x)
            disp_preds.append(disp_up if test_mode else disp_up.detach())
        if self.args.use_camera_geometry:
            fx = cam_L["K"][:,0,0].view(B,1,1,1)
            baseline = cam_L["baseline"].view(B,1,1,1)

            depth_pred = (fx * baseline) / (disp_up + 1e-6)

            # 3D points in world frame
            C_L = -torch.bmm(
                cam_L["R"].transpose(1,2),
                cam_L["t"].unsqueeze(-1)
            ).squeeze(-1)                        # [B,3]
            C_L = C_L.view(B,3,1,1)

            _, _, H_depth, W_depth = depth_pred.shape
            _, _, H_rays, W_rays = rays_L.shape
            
            if H_rays != H_depth or W_rays != W_depth:
                rays_L = F.interpolate(
                    rays_L, 
                    size=(H_depth, W_depth), 
                    mode='bilinear', 
                    align_corners=False
                )
            X = (C_L + depth_pred * rays_L).permute(0,2,3,1)
        else:
            depth_pred = None
            X = None
        if test_mode:
            return {
                'disp_pred': disp_up,
                'depth_pred': depth_pred,
                'points_3d': X
            }

        return {
            'init_disp': init_disp,
            'disp_preds': disp_preds,
            'disp_pred': disp_up,
            'depth_pred': depth_pred,
            'points_3d': X
        }


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
        B,_,H,W = image1.shape
        img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
        img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
        padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
        img1_small, img2_small = padder.pad(img1_small, img2_small)
        disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
        disp_small = padder.unpad(disp_small.float())
        disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
        disp_small_up = disp_small_up.clip(0, None)

        padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
        image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
        disp_small_up += padder._pad[0]
        init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
        disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
        disp = padder.unpad(disp.float())
        return disp
    
    def get_loss(self, model_pred, input_data):
        loss, loss_info = self.criterion(model_pred, input_data)
        loss_info = {'scalar/train/total_loss': loss.item(),
                     'scalar/train/loss_reproj': loss_info['scalar/loss_reproj'],
                     'scalar/train/loss_3d': loss_info['scalar/loss_3d']}
        return loss, loss_info