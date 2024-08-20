"""
This file contains helper functions for rendering and batching rays. These functions are designed to handle large inputs in smaller chunks to avoid out-of-memory errors.
Until now, most of these functions are a translation of the Pytorch version of NeRF codebase, with some modifications.
Take a look here: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py
"""
import numpy as np
import torch
import torch.nn.functional as F


def batchify(inputs, fn, chunk):
    """Render rays in smaller minibatches to avoid OOM."""
    return torch.cat(
        [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)],
        0,
    )


def run_network(inputs, model, netchunk):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    outputs_flat = batchify(inputs_flat, model, netchunk)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def raw2outputs(raw, z_vals, rays_d, device):
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1
    )  # [n_rays, n_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    alpha = raw2alpha(raw[..., 3], dists)  # [n_rays, n_samples]
    weights = (
        alpha
        * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1), device=device), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )[:, :-1]
    )

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]
    return rgb_map


def render_rays(ray_batch, model, n_samples, chunk, device):
    DEBUG = True
    n_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [n_rays, 3] each
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([n_rays, n_samples])

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [n_rays, n_samples, 3]

    raw = run_network(pts, model, chunk)
    rgb_map = raw2outputs(raw, z_vals, rays_d, device)

    if (torch.isnan(rgb_map).any() or torch.isinf(rgb_map).any()) and DEBUG:
        print("[Numerical Error] rgb_map contains nan or inf.")

    return rgb_map


def batchify_rays(rays_flat, model, chunk, device):
    """Render rays in smaller minibatches to avoid OOM."""
    return torch.cat(
        [
            render_rays(rays_flat[i : i + chunk], model, 16, chunk, device)
            for i in range(0, rays_flat.shape[0], chunk)
        ],
        0,
    )


def get_rays(h, w, intrinsic_mat, camera_to_world_mat, device):
    i, j = torch.meshgrid(
        torch.linspace(0, w - 1, w, device=device),
        torch.linspace(0, h - 1, h, device=device),
        indexing="ij",
    )
    i, j = i.t(), j.t()
    dirs = torch.stack(
        [
            (i - intrinsic_mat[0][2]) / intrinsic_mat[0][0],
            -(j - intrinsic_mat[1][2]) / intrinsic_mat[1][1],
            -torch.ones_like(i),
        ],
        -1,
    )

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * camera_to_world_mat[:3, :3], -1
    )  # dot product, equals to: [camera_to_world_mat.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = camera_to_world_mat[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def render(
    model,
    h,
    w,
    intrinsic_mat,
    chunk,
    device,
    rays=None,
    camera_to_world_mat=None,
    near=0.0,
    far=1.0,
):
    if camera_to_world_mat is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(h, w, intrinsic_mat, camera_to_world_mat, device)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    batch_shape = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = (
        near * torch.ones_like(rays_d[..., :1]),
        far * torch.ones_like(rays_d[..., :1]),
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # Render and reshape
    rgb_map = batchify_rays(rays, model, chunk, device)
    print(f"~= { rgb_map.shape = }")
    map_shape = list(batch_shape[:-1]) + list(rgb_map.shape[1:])
    rgb_map = torch.reshape(rgb_map, map_shape)
    print(f"@= { rgb_map.shape = }")

    return rgb_map
