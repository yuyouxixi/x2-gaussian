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
import sys
import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
import time as timeku

sys.path.append("./")
from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.dataset.cameras import Camera
from x2_gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    time=0.1,
    stage='fine',
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling
        rotations = pc._rotation

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        means3D_final, scales_final, rotations_final = pc._deformation(means3D, scales, rotations, density, time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    vol_pred, radii = voxelizer(
        means3D=means3D_final,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    stage='fine',
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling
        rotations = pc._rotation

    # neg_inf_mask = torch.isinf(scales)
    # if neg_inf_mask.sum() > 0:
    #     print('scales inf !!!')

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        # breakpoint()
        means3D_final, scales_final, rotations_final = pc._deformation(means3D, scales, rotations, density, time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria. 
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

def render_prior_oneT(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    stage='fine',
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    period=pc.period
    period = torch.exp(period)
    range_max=torch.tensor((60.0)).cuda()
    time = time * range_max
    with torch.no_grad():
        num_periods = int(range_max / period)
        cur_period_num = int(time[0] / period)
        period_list = list(range(num_periods))
        relative_indices = [i - cur_period_num for i in period_list if i != cur_period_num]
        if 1 in relative_indices:
            sampled_offset = 1
        elif -1 in relative_indices:
            sampled_offset = -1
        else:
            breakpoint()

    new_time = time + torch.tensor(sampled_offset).to(means3D.device) * period
    time = new_time / range_max


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling
        rotations = pc._rotation

    # neg_inf_mask = torch.isinf(scales)
    # if neg_inf_mask.sum() > 0:
    #     print('scales inf !!!')

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        # breakpoint()
        means3D_final, scales_final, rotations_final = pc._deformation(means3D, scales, rotations, density, time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    # breakpoint()

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria. 
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }