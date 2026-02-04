# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import sys

# Ensure dinov2 module can be imported by adding this directory to sys.path
# This is needed when hubconf.py is loaded via torch.hub.load
_hubconf_dir = os.path.dirname(os.path.abspath(__file__))
if _hubconf_dir not in sys.path:
    sys.path.insert(0, _hubconf_dir)

from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg
from dinov2.hub.classifiers import dinov2_vitb14_lc, dinov2_vitg14_lc, dinov2_vitl14_lc, dinov2_vits14_lc
from dinov2.hub.classifiers import dinov2_vitb14_reg_lc, dinov2_vitg14_reg_lc, dinov2_vitl14_reg_lc, dinov2_vits14_reg_lc
from dinov2.hub.depthers import dinov2_vitb14_ld, dinov2_vitg14_ld, dinov2_vitl14_ld, dinov2_vits14_ld
from dinov2.hub.depthers import dinov2_vitb14_dd, dinov2_vitg14_dd, dinov2_vitl14_dd, dinov2_vits14_dd
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l

dependencies = ["torch"]
