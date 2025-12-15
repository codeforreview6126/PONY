
import h5py
import torch
import numpy as np
from data_util.pre_processing import encode_gas

def upsample_single(pc, upsample_points):
    """
    Upsample a single point cloud tensor of shape (1, C, N) to (1, C, upsample_points).
    """
    _, C, N = pc.shape
    pc = pc.squeeze(0)  # (C, N)

    if N == upsample_points:
        return pc.unsqueeze(0)

    repeat_factor = upsample_points // N
    remainder = upsample_points % N

    repeated = []

    if repeat_factor > 0:
        total_N = repeat_factor * N
        rand_idx = torch.randint(low=0, high=N, size=(total_N,), device=pc.device)
        repeated_pc = pc[:, rand_idx]
        noise = 1e-2 * torch.randn_like(repeated_pc)
        repeated_pc += noise
        repeated.append(repeated_pc)

    if remainder > 0:
        idx = torch.randint(low=0, high=N, size=(remainder,), device=pc.device)
        sampled = pc[:, idx]
        noise = 1e-2 * torch.randn_like(sampled)
        sampled += noise
        repeated.append(sampled)

    pc_new = torch.cat(repeated, dim=1) if repeated else pc  # (C, upsample_points)
    return pc_new.unsqueeze(0)  # (1, C, upsample_points)

def prepare_single_sample(
    mof_id,
    gas,
    T,
    P,
    forcefield,
    hdf5_path,
    max_abs_charge,
    max_abs_eps,
    max_abs_sigma,
    T_scale_coef,
    P_scale_coef,
    log_P_scale_coef,
    log_scaling_P=True,
    upsample_points=2048  # Desired number of points
):
    """
    Prepare a single point cloud tensor for a given MOF ID and conditions,
    where each point has the original features + duplicated global condition features.
    """
    mof_id = str(mof_id)
    with h5py.File(hdf5_path, "r") as f:
        if mof_id not in f:
            raise ValueError(f"MOF ID {mof_id} not found in {hdf5_path}")
        pc_array = f[mof_id][()][:, :6]  # (N_points, 6)

    # Normalize feature channels
    pc_array[:, 3] /= max_abs_charge
    pc_array[:, 4] /= max_abs_eps
    pc_array[:, 5] /= max_abs_sigma

    # Convert to (1, 6, N) torch tensor
    pc_tensor = torch.tensor(pc_array.T, dtype=torch.float32).unsqueeze(0)  # (1, 6, N)

    # === Upsample to user-defined number of points ===
    pc_tensor = upsample_single(pc_tensor, upsample_points)  # still (1, 6, N)

    # Gas + condition encoding
    gas_encoded = torch.tensor(encode_gas(gas, forcefield), dtype=torch.float32)  # (G,)
    T_tensor = torch.tensor([T / T_scale_coef], dtype=torch.float32)              # (1,)
    P_tensor = torch.tensor(
        [np.log10(P) / log_P_scale_coef] if log_scaling_P else [P / P_scale_coef],
        dtype=torch.float32
    )  # (1,)

    # Combine extra features: (E,)
    extra_features = torch.cat([gas_encoded, T_tensor, P_tensor], dim=0)  # (E,)
    
    return pc_tensor, extra_features

import numpy as np

def assign_gradcam_importance(mof_id, hdf5_path, 
                              grad_cam, max_noise=0.1,
                              normalize=True):
    
    # Step 0: Load MOF as Point Cloud
    with h5py.File(hdf5_path, "r") as f:
        if str(mof_id) not in f:
            raise ValueError(f"MOF ID {mof_id} not found in {hdf5_path}")
        point_cloud = f[str(mof_id)][()]  # (N_points, 6)

    # Step 1: Merge close points in grad_cam
    gc_points = grad_cam[:, :3]  # xyz
    gc_values = grad_cam[:, 3]   # i values

    merged = np.zeros(len(gc_points), dtype=bool)
    merged_gc = []

    for i in range(len(gc_points)):
        if merged[i]:
            continue
        dists = np.linalg.norm(gc_points - gc_points[i], axis=1)
        close_idx = np.where(dists < max_noise)[0]
        total_i = gc_values[close_idx].sum()
        merged[i] = True
        merged[close_idx] = True
        merged_gc.append(np.append(gc_points[i], total_i))

    merged_gc = np.array(merged_gc)  # (M_merged,4)

    # Step 2: Add importance column to point_cloud
    N = point_cloud.shape[0]
    importance_col = np.full((N, 1), -1.0)  # default -1

    for idx in range(N):
        pc_xyz = point_cloud[idx, :3]
        if len(merged_gc) > 0:
            dists = np.linalg.norm(merged_gc[:, :3] - pc_xyz, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_noise:
                importance_col[idx, 0] = merged_gc[min_idx, 3]

    # Step 3: Normalize importance values (excluding -1)
    if normalize:
        mask = importance_col[:, 0] != -1
        if np.any(mask):
            values = importance_col[mask, 0]
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:  # avoid divide by zero
                importance_col[mask, 0] = (values - min_val) / (max_val - min_val)
            else:
                importance_col[mask, 0] = 1.0  # all values equal -> set 1

    # Append as new column
    point_cloud_with_importance = np.hstack([point_cloud, importance_col])

    return point_cloud_with_importance

def normalize_importance_batch(point_clouds, global_max=None):
    
    # Step 1: Find global maximum if not provided
    if global_max is None:
        all_importance = []
        for pc in point_clouds:
            importance_vals = pc[:, -1]
            valid = importance_vals[importance_vals != -1]
            if valid.size > 0:
                all_importance.append(valid.max())
        if len(all_importance) == 0:
            raise ValueError("No valid importance values found for normalization.")
        global_max = max(all_importance)

    # Step 2: Normalize each point cloud
    normalized_pcs = []
    for pc in point_clouds:
        pc_copy = pc.copy()
        mask = pc_copy[:, -1] != -1
        pc_copy[mask, -1] = pc_copy[mask, -1] / global_max
        normalized_pcs.append(pc_copy)

    return normalized_pcs