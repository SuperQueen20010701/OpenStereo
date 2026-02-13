import torch
import numpy as np

class CameraGeometryAugmentation:
    def __init__(
        self,
        f_range=(700, 1400),
        baseline_range=(0.1, 1.2),
        cx_jitter=10.0,
        cy_jitter=10.0,
        canonical_f=1050.0,
        canonical_baseline=0.54,
        eps=1e-6
    ):
        self.f_range = f_range
        self.baseline_range = baseline_range
        self.cx_jitter = cx_jitter
        self.cy_jitter = cy_jitter
        self.f0 = canonical_f
        self.B0 = canonical_baseline
        self.eps = eps

    def __call__(self, sample):
        """
        sample:
          left:  [H,W,3]
          right: [H,W,3]
          disp:  [H,W]
        """
        disp = sample['disp']  # disp in tensor
        # Convert to numpy if it's a Tensor
        if torch.is_tensor(disp):
            disp = disp.cpu().numpy() # disp in numpy array
        # Ensure disp is numpy array with float32 dtype
        disp = np.asarray(disp, dtype=np.float32)
        H, W = disp.shape

        depth = self.f0 * self.B0 / (disp + self.eps)
        depth = np.asarray(depth, dtype=np.float32)

        f = np.random.uniform(*self.f_range)
        B = np.random.uniform(*self.baseline_range)

        cx = W / 2 + np.random.uniform(-self.cx_jitter, self.cx_jitter)
        cy = H / 2 + np.random.uniform(-self.cy_jitter, self.cy_jitter)

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # stereo: rectified
        R_L = np.eye(3, dtype=np.float32)
        R_R = np.eye(3, dtype=np.float32)
        t_L = np.zeros(3, dtype=np.float32)
        t_R = np.array([B, 0, 0], dtype=np.float32)

        xs, ys = np.meshgrid(
            np.arange(W),
            np.arange(H)
        )

        X = (xs - cx) / f * depth
        Y = (ys - cy) / f * depth
        Z = depth

        uL = f * X / Z + cx
        uR = f * (X - B) / Z + cx

        disp_new = uL - uR
        disp_new = np.clip(disp_new, 0, W)

        # ---------------------------
        # 4. write back
        # ---------------------------
        sample['disp'] = disp_new.astype(np.float32)
        sample['camera'] = {
            'K_L': K,
            'K_R': K.copy(),
            'R_L': R_L,
            'R_R': R_R,
            't_L': t_L,
            't_R': t_R,
            'baseline': np.array([B], dtype=np.float32)
        }

        return sample
