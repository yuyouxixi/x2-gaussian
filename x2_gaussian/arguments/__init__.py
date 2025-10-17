#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from x2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min =  0.0  # percent of volume size  0.0005
        self.scale_max =  0.5  # percent of volume size
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000

        self.deformation_lr_init = 0.0002     # 0.0002   4
        self.deformation_lr_final = 0.00002     # 0.00002   4
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.002  # 0.002   4
        self.grid_lr_final = 0.0002   # 0.0002   4

        self.period_lr_init = 0.0002
        self.period_lr_final = 0.00002
        self.period_lr_max_steps = 30_000

        self.deformation_low_lr_init = 0.0002     # 0.0002   4
        self.deformation_low_lr_final = 0.00002     # 0.00002   4
        self.deformation_low_lr_delay_mult = 0.01
        self.grid_low_lr_init = 0.002  # 0.002   4
        self.grid_low_lr_final = 0.0002   # 0.0002   4

        self.deformation_high_lr_init = 0.0002     # 0.0002   4
        self.deformation_high_lr_final = 0.00002     # 0.00002   4
        self.deformation_high_lr_delay_mult = 0.01
        self.grid_high_lr_init = 0.002  # 0.002   4
        self.grid_high_lr_final = 0.0002   # 0.0002   4

        self.hf_weights_lr_init = 0.0002
        self.hf_weights_lr_final = 0.00002
        self.hf_weights_lr_max_steps = 30_000

        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.lambda_prior = 1.0
        self.lambda_prior_3d = 0.01 # useless
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 500_000
        super().__init__(parser, "Optimization Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 #64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.density_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.001 # TV loss of temporal grid  0.01
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.period_regulation_weight = 1.0   # useless
        self.period_construction_weight = 1e-5  # useless
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,   # 32
                             'resolution': [64, 64, 64, 150]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }    # 150
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity     # True
        # self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        
        super().__init__(parser, "ModelHiddenParams")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
