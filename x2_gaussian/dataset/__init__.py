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
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from x2_gaussian.gaussian import GaussianModel
from x2_gaussian.arguments import ModelParams
from x2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from x2_gaussian.utils.camera_utils import cameraList_from_camInfos
from x2_gaussian.utils.general_utils import t2a


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)


        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc, stage):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply

        self.gaussians.save_deformation(point_cloud_path)

        if queryfunc is not None:
            breath_cycle = 3.0  # 呼吸周期
            num_phases = 10  # 相位数
            phase_time = breath_cycle / num_phases
            mid_phase_time = phase_time / 2
            scanTime = 60.0
            for t in range(10):
                time = (mid_phase_time + phase_time * t) / scanTime

                vol_pred = queryfunc(self.gaussians, time, stage)["vol"]
                vol_gt = self.vol_gt[t]
                np.save(osp.join(point_cloud_path, "vol_gt_T" + str(t) + ".npy"), t2a(vol_gt))
                np.save(
                    osp.join(point_cloud_path, "vol_pred_T" + str(t) + ".npy"),
                    t2a(vol_pred),
                )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras


def to_cam(pt_world):
    v = np.array([*pt_world, 1.0], dtype=np.float32)
    cam = v @ WV.T
    return cam[:3] / cam[3]

def project_point(world_pt):
    X = torch.tensor([world_pt[0], world_pt[1], world_pt[2], 1.0], dtype=torch.float32, device=cam.world_view_transform.device)
    clip = X @ cam.world_view_transform.T @ cam.projection_matrix.T
    ndc = clip[:3] / torch.clamp(clip[3], min=1e-8)
    px = (ndc[0].item() * 0.5 + 0.5) * W
    py = (ndc[1].item() * 0.5 + 0.5) * H
    return px, py