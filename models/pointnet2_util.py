import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
    dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    reshape_shape = list(idx.shape)
    reshape_shape[1:] = [1] * (len(reshape_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).reshape(reshape_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# FIXED FUNCTION (Returns Query Point Itself):
def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute pairwise squared distances
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    # Sort distances and get indices
    sorted_dists, group_idx = sqrdists.sort(dim=-1)  # both [B, S, N]

    # Mask out points beyond radius
    mask = sorted_dists > radius ** 2
    group_idx[mask] = N

    # Keep only nsample neighbors
    group_idx = group_idx[:, :, :nsample]   # [B, S, nsample]

    # Determine first neighbor in each group
    group_first = group_idx[:, :, 0]        # [B, S]
    empty_mask = group_first == N           # True if no valid neighbors for that group

    # For empty groups, use the query point itself as the neighbor
    # Create indices [0, 1, 2, ..., S-1] for each batch
    self_idx = torch.arange(S, device=device).reshape(1, S).repeat(B, 1)  # [B, S]
    group_first[empty_mask] = self_idx[empty_mask]

    # Repeat first index to fill the entire group
    group_first_expanded = group_first.reshape(B, S, 1).repeat(1, 1, nsample)  # [B, S, nsample]

    # Replace any N indices with the first valid index
    mask = group_idx == N
    group_idx[mask] = group_first_expanded[mask]

    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    # return FPS indices when requested (for Grad-CAM)
    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            fps_idx = None
        else:
            new_xyz, new_points, fps_idx = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, returnfps=True
            )
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points, fps_idx

    
""" # This code has been adapted from:
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
"""