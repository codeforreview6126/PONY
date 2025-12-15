import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# AUGMENTATIONS ONLY KEPT FOR TRAINING SET...!
def train_val_test_split(
    df,
    id_column='ID',
    train_size=0.40,
    val_size=0.10,
    test_size=0.10,
    random_state=42,
    adsorption_model=False
):
    # Extract base IDs (everything before the last underscore)
    base_ids = df[id_column].unique()
    n_unique = len(base_ids)

    print(f"Total unique base IDs found: {n_unique}")
    print(f"Base IDs: {base_ids[:10]}{'...' if len(base_ids) > 10 else ''}")

    # Validate ratios
    total_ratio = train_size + val_size + test_size
    if total_ratio > 1.0:
        raise ValueError("train_size + val_size + test_size must be <= 1.0")

    # Count per split
    n_test = int(test_size * n_unique)
    n_val = int(val_size * n_unique)
    n_train = int(train_size * n_unique)

    # RNG
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    shuffled_ids = rng.permutation(base_ids)

    test_ids = shuffled_ids[:n_test]
    val_ids = shuffled_ids[n_test:n_test + n_val]
    train_ids = shuffled_ids[n_test + n_val:n_test + n_val + n_train]
    
    train_df = df[df[id_column].isin(train_ids)].copy()
    val_df = df[df[id_column].isin(val_ids)].copy()
    test_df = df[df[id_column].isin(test_ids)].copy()
    
    # Get final IDs
    train_ids = train_df[id_column].tolist()
    val_ids = val_df[id_column].tolist()
    test_ids = test_df[id_column].tolist()

    print(f"Number of train samples: {len(train_ids)}")
    print(f"Number of val samples: {len(val_ids)}")
    print(f"Number of test samples: {len(test_ids)}")

    # Scaling placeholders or scaling coefficients for adsorption data
    if adsorption_model == True:
        T_scale_coef = train_df["T"].max()
        P_scale_coef = train_df["P"].max()
        log_P = np.log10(train_df["P"])
        log_P_scale_coef = log_P.abs().max()
    else:
        T_scale_coef = 1
        P_scale_coef = 1
        log_P_scale_coef = 1

    return train_df, val_df, test_df, train_ids, val_ids, test_ids, T_scale_coef, P_scale_coef, log_P_scale_coef

def train_val_test_split_protein(
    df,
    id_column='ID',
    train_size=0.40,
    val_size=0.10,
    test_size=0.10,
    random_state=42,
    adsorption_model=False,
    use_upsample=True
):
    # Extract base IDs (everything before the last underscore)
    base_ids = df[id_column].str.split("_").str[0].unique()
    n_unique = len(base_ids)

    print(f"Total unique base IDs found: {n_unique}")
    print(f"Base IDs: {base_ids[:10]}{'...' if len(base_ids) > 10 else ''}")

    # Validate ratios
    total_ratio = train_size + val_size + test_size
    if total_ratio > 1.0:
        raise ValueError("train_size + val_size + test_size must be <= 1.0")

    # Count per split
    n_test = int(test_size * n_unique)
    n_val = int(val_size * n_unique)
    n_train = int(train_size * n_unique)

    # RNG
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    shuffled_ids = rng.permutation(base_ids)

    test_base_ids = shuffled_ids[:n_test]
    val_base_ids = shuffled_ids[n_test:n_test + n_val]
    train_base_ids = shuffled_ids[n_test + n_val:n_test + n_val + n_train]
    
    # Decide to use upsampled training set or not. 
    if use_upsample == True:
        train_df = df[df[id_column].str.split("_").str[0].isin(train_base_ids)].copy() # Training set: all augmentations
    else:
        train_df = df[                                                                 # Training set: only original point clouds
            df[id_column].str.split("_").str[0].isin(train_base_ids) &
            df[id_column].str.endswith("_0")
        ].copy()
    
    # Validation and test sets: only _0 versions
    val_df = df[
        df[id_column].str.split("_").str[0].isin(val_base_ids) &
        df[id_column].str.endswith("_0")
    ].copy()

    test_df = df[
        df[id_column].str.split("_").str[0].isin(test_base_ids) &
        df[id_column].str.endswith("_0")
    ].copy()
    
    # Get final IDs
    train_ids = train_df[id_column].tolist()
    val_ids = val_df[id_column].tolist()
    test_ids = test_df[id_column].tolist()

    print(f"Number of train samples: {len(train_ids)}")
    print(f"Number of val samples: {len(val_ids)}")
    print(f"Number of test samples: {len(test_ids)}")

    # Scaling placeholders or scaling coefficients for adsorption data
    if adsorption_model == True:
        T_scale_coef = train_df["T"].max()
        P_scale_coef = train_df["P"].max()
        log_P = np.log10(train_df["P"])
        log_P_scale_coef = log_P.abs().max()
    else:
        T_scale_coef = 1
        P_scale_coef = 1
        log_P_scale_coef = 1

    return train_df, val_df, test_df, train_ids, val_ids, test_ids, T_scale_coef, P_scale_coef, log_P_scale_coef


import h5py
import numpy as np
from tqdm import tqdm

def find_scaling_coefficients(
    train_ids,              # List of unique MOF IDs in training set
    h5_path,                # Path to the HDF5 file
    num_atomic_features=4,  # Number of additional feature columns after x, y, z
):

    atomic_feature_scaling_coef = np.zeros(num_atomic_features, dtype=float)

    with h5py.File(h5_path, "r") as f:
        for mof_id in tqdm(train_ids, desc="Scanning point clouds in HDF5"):
            if str(mof_id) not in f:
                print(f"WARNING: MOF ID {mof_id} not found in HDF5 file.")
                continue

            # Load only needed columns: x,y,z + feature columns
            pc_array = f[str(mof_id)][()]
            feature_array = pc_array[:, 3:3 + num_atomic_features]

            # Update per-feature max absolute values
            current_max = np.abs(feature_array).max(axis=0)
            atomic_feature_scaling_coef = np.maximum(atomic_feature_scaling_coef, current_max)

    return atomic_feature_scaling_coef.tolist()


def encode_gas(gas_name, forcefield):

    # Determine which H2 Model:
    if gas_name == "H2" and forcefield == "TraPPE-H2-3SM":
        gas_name_plus_ff = "H2_Sun"
    elif gas_name == "H2" and forcefield == "Darkrim-Levesque":
        gas_name_plus_ff = "H2_DL"
    elif gas_name == "H2" and forcefield == "Michels-Degraaff-Tenseldam with Darkrim-Levesque partial charges":
        gas_name_plus_ff = "H2_DL"
    elif gas_name == 1.0:   # default to methane if gas_name is arbitrary - will be disregarded. 
        gas_name_plus_ff = "CH4"
    else:
        gas_name_plus_ff = gas_name

    phys_molecular_model_descriptors = {
        
        # CH4, N2, CO2, H2 Adsorbate Descriptors (MaxAbs Scaled) - Hard-coding is acceptable since all molecules across splits.
        "CH4"    : [0.364539072,0.626591023,0.623425915,0.050995802,0.418644381,0.022545805,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0],

        "N2"     : [0.636506169,0.414930283,0.460303905,0.166115924,0.291485825,0.024165863,0.489266547,1,0.7,-1,0.243243243,0.887399464,1,0.972222222,1,0,0,1,1,-1,0.455696203,1],

        "CO2"    : [1,1,1,1,1,1,0.930232558,1,0.68,-0.726141079,0.533783784,0.81769437,1,1,0.726141079,0.735694823,0.946585531,1,0.971428571,-0.726141079,1,0.921450151],

        "H2_DL"  : [0.045808812,0.10913161,0.178249495,-0.955613111,0.064456552,0.013905497,0.331842576,1,0.7,0.970954357,0,0,1,0.972222222,-0.970954357,1,1,1,1,0.970954357,0,0],

        "H2_Sun" : [0.045808812,0.10913161,0.178249495,-0.955613111,0.064456552,0.013905497,0.331842576,1,0.7,0.970954357,0,0,1,0.972222222,-0.970954357,1,1,1,1,0.970954357,0,0],
                    
        "Ar"     : [0.907768865,0.664135593,0.496054449,0.0041841,0,0.448382126,0.386924,0.679300828,0.665194007,0,1,0.376388889,0.748246844,0.724342782,0.676041296,
                    0,0,0.777027027,0.913404826,0,0,0,0,0,0,0,0]
    }
    
    vec = phys_molecular_model_descriptors[gas_name_plus_ff]

    return vec

def prepare_data(
    data_df,
    hdf5_path,
    num_atomic_features,
    atomic_feature_scaling_coef,
    T_scale_coef, 
    P_scale_coef,
    log_P_scale_coef,
    log_scaling_P=True,
    target_col = "Uptake(mol/kg)",
    mode = 0
):
    data_samples = []
    
    with h5py.File(hdf5_path, "r") as f:
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preparing samples"):

            mof_id = str(row["ID"])
            if mof_id not in f:
                print(f"WARNING: MOF {mof_id} not found in HDF5.")
                continue

            if mode == 2:   # single task
                gas = 1.0
                T = 1.0
                P = 1.0
                forcefield = 1.0
            else: 
                gas = row["Gas"]
                T = row["T"]
                P = row["P"]
                forcefield = row["Forcefield"]

            target = row[target_col]
            
            pc_array = f[mof_id][()][:, :3 + num_atomic_features]

            # Scale using atomic feature coefficients
            for i in range(3, 3 + num_atomic_features):
                pc_array[:, i] /= atomic_feature_scaling_coef[i - 3]
            
            # Convert to tensor → (6, N_points)
            pc_tensor = torch.tensor(pc_array.T, dtype=torch.float32)

            # Encode Gas
            gas_encoded = torch.tensor(encode_gas(gas, forcefield), dtype=torch.float32)

            # Prepare extra features
            T_tensor = torch.tensor([T / T_scale_coef], dtype=torch.float32)
            P_tensor = torch.tensor(
                [np.log10(P) / log_P_scale_coef] if log_scaling_P else [P / P_scale_coef], dtype=torch.float32
            )
            
            if mode == 0:
                extra_features_tensor = torch.cat([gas_encoded, T_tensor, P_tensor], dim=0)
            elif mode == 1: 
                extra_features_tensor = torch.cat([T_tensor, P_tensor], dim=0)
            elif mode == 2:
                extra_features_tensor = None
            else: 
                raise ValueError("Invalid mode selected. Please choose one of 0, 1, 2 for extra feature configuration.")

            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            data_samples.append((pc_tensor, extra_features_tensor, target_tensor, idx))
    
    print(f"Total samples prepared: {len(data_samples)}")
    return data_samples

def upsample_collate(batch, drop_points=True):

    # Hyperparameters
    xyz_noise_std = 0.25
    feat_noise_std = 0.005
    min_points = 1024
    drop_fraction = 0.1

    pcs, extras, targets, indices = zip(*batch)

    # Find max number of points across batch (before dropping)
    max_points_batch = max(pc.shape[1] for pc in pcs)
    max_points = max(max_points_batch, min_points)

    upsampled_pcs = []

    for pc in pcs:
        C, N = pc.shape

        # Step 1: Drop random fraction of points (if enabled)
        if drop_points:
            num_points_to_drop = int(round(drop_fraction * N))
            if num_points_to_drop > 0:
                keep_idx = torch.randperm(N, device=pc.device)[num_points_to_drop:]
                pc = pc[:, keep_idx]
                C, N = pc.shape  # update shape after dropping

        # Step 2: Upsample to max_points
        num_points_to_add = max_points - N
        if num_points_to_add > 0:
            idx = torch.randint(0, N, (num_points_to_add,), device=pc.device)
            extra_points = pc[:, idx]
            pc_new = torch.cat([pc, extra_points], dim=1)
        else:
            pc_new = pc

        # Step 3: Add noise separately
        xyz_noise = xyz_noise_std * torch.randn_like(pc_new[:3, :])
        pc_new[:3, :] += xyz_noise

        if C > 3:
            feat_noise = feat_noise_std * torch.randn_like(pc_new[3:, :])
            pc_new[3:, :] += feat_noise

        upsampled_pcs.append(pc_new)

    pcs_tensor = torch.stack(upsampled_pcs, dim=0)

    extra_feat_dim = batch[0][1].shape[0] if batch[0][1] is not None else 0
    extras = [
        e if e is not None else torch.zeros(extra_feat_dim, device=pcs_tensor.device)
        for e in extras
    ]
    extras_tensor = torch.stack(extras, dim=0)

    targets_tensor = torch.stack(targets, dim=0).squeeze()
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    return pcs_tensor, extras_tensor, targets_tensor, indices_tensor