import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models.pointnet2_v2_film_clipped import get_model, get_loss
from training_util.training_util_publication import MOFDataset, train_one_epoch, evaluate
from data_util.pre_processing_v2 import train_val_test_split, find_scaling_coefficients, prepare_data, upsample_collate
from data_util.calculate_stats import compute_r2, compute_mae, compute_rmse
from data_util.plotting import plot_learning_curve, parity_plot
 
### CONFIGURE TRAINING:
### Model checkpoining with highest RMSE is configured as default. 

extra_feat_dim                  =  # global descriptor dimension (set as 24 for hMOFX-DB and 0 for rest)
mode_num                        =  # this program was originally designed for benchmark on hMOFX-DB only. set this value to 0 for hMOFX-DB and 2 for all other applications. 
ads_model                       =  # this program was originally designed for benchmark on hMOFX-DB only. set this value to True for hMOFX-DB and False for all other applications (even CoRE-CO2 and hMOF-CO2). 
num_cpus                        = 12
learning_rate                   = 1e-3
EPOCHS                          = 100 # Refer to SI
num_atomic_features             = 8 # 8 for all other applications 9 for protein-ligand binding affinity
target_name                     = "" # column name containing target variables from target CSV file
structure_id_col                = "ID" # column name containing structural ID (usually 'ID')

h5_path                         = "" # HDF5 file containing atomic point clouds across training, validation, and test structures. 
train_df_path                   = ""
val_df_path                     = ""
test_data_path                  = ""

model_file_name                 = "{working directory for storing all data}/best_model.pth"
train_csv_name                  = "{working directory for storing all data}/train_data.csv"
val_csv_name                    = "{working directory for storing all data}/validation_data.csv"
test_csv_name                   = "{working directory for storing all data}/test_data.csv"

learning_curve_df_path          = "{working directory for storing all data}/learning_curve_data.csv"
learning_curve_path             = "{working directory for storing all data}/learning_curve.png"
parity_plot_path                = "{working directory for storing all data}/validation_parity_plot.png"
test_parity_plot_path           = "{working directory for storing all data}/test_parity_plot.png"

### FOR LEARNING CURVE:
training_loss = []
validation_loss = []

# Keep as true (legacy)
log_P  = True

if __name__ == "__main__":

    ####### Load Device 
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ####### Prepare Training/Validation Data 
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)

    train_df, _, _, train_ids, _, _, T_scale_coef, P_scale_coef, log_P_scale_coef = train_val_test_split(
        df           = train_df,
        id_column    = structure_id_col,
        train_size   = 1.00,
        val_size     = 0.00,
        test_size    = 0.00, 
        random_state = 42,
        adsorption_model = ads_model
    )
    
    _, val_df, _, _, val_ids, _, _, _, _ = train_val_test_split(
        df           = val_df,
        id_column    = structure_id_col,
        train_size   = 0.00,
        val_size     = 1.00,
        test_size    = 0.00, 
        random_state = 42,
        adsorption_model = ads_model
    )

    atomic_feature_scaling_coef = find_scaling_coefficients(
        train_ids, h5_path, num_atomic_features
    )

    train_samples = prepare_data(train_df, h5_path,
                                 num_atomic_features, atomic_feature_scaling_coef, 
                                 T_scale_coef, P_scale_coef, log_P_scale_coef,
                                 log_scaling_P = log_P, target_col = target_name, mode = mode_num
                                 )
    
    val_samples = prepare_data(val_df, h5_path,
                              num_atomic_features, atomic_feature_scaling_coef, 
                              T_scale_coef, P_scale_coef, log_P_scale_coef,
                              log_scaling_P = log_P, target_col = target_name, mode = mode_num
                              )

    # Print shape for train_sample list:
    print("Outer length:", len(train_samples))
    print("Inner length:", len(train_samples[0]) if len(train_samples) > 0 else "Empty")

    train_dataset = MOFDataset(train_samples)
    val_dataset = MOFDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=150, 
        shuffle=True, 
        num_workers=num_cpus, 
        collate_fn=lambda batch: upsample_collate(batch, drop_points=True)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=50, 
        shuffle=False, 
        num_workers=num_cpus, 
        collate_fn=lambda batch: upsample_collate(batch, drop_points=False)
    )
    
    ####### Print Scaling Coefficients for Manual Setup
    print("---- Scaling Coefficients for Point Cloud -------")
    print("Scaling Coefficients  : " + str(atomic_feature_scaling_coef))
    
    print("---- Scaling Coefficients for Extra Features ----")
    print("Maximum Temperature   : " + str(T_scale_coef))
    print("Maximum Pressure      : " + str(P_scale_coef))
    print("Maximum log(Pressure) : " + str(log_P_scale_coef))


    # ####### Load Model (Pointnet++ Single Scale Grouping + FiLM Modulation)
    model = get_model(extra_feat_dim=extra_feat_dim, atomic_feature_channel_num=num_atomic_features).to(device)
    print(model)

    loss_fn = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)  # decay LR by 0.1 every 20 epochs

    ###### Training Loop
    best_val_rmse = 9999
    
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_preds, train_targets, train_row_indicies = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_targets, val_row_indicies = evaluate(model, val_loader, loss_fn, device)
        val_rmse = compute_rmse(val_targets, val_preds)

        # Save model/predictions with lowest MAE on validation set
        if val_rmse < best_val_rmse and not np.isnan(val_rmse):
            best_val_rmse = val_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), model_file_name)
            
            # Save Validation Target Predictions + Metadata
            valid_indices = val_df.index.values
            valid_mask = np.isin(val_row_indicies, valid_indices)
            safe_indices = val_row_indicies[valid_mask]
            metadata_df = val_df.loc[safe_indices].reset_index(drop=True)

            val_target_df = pd.DataFrame({
                'val_true': val_targets,
                'val_pred': val_preds
            })

            combined_val_df = pd.concat([metadata_df, val_target_df], axis=1)

            combined_val_df.to_csv(val_csv_name, index=False)

            # Save Training Target Predictions + Metadata
            valid_indices_train = train_df.index.values
            valid_mask_train = np.isin(train_row_indicies, valid_indices_train)
            safe_indices_train = train_row_indicies[valid_mask_train]
            metadata_df_train = train_df.loc[safe_indices_train].reset_index(drop=True)

            train_target_df = pd.DataFrame({
                'train_true': train_targets,
                'train_pred': train_preds
            })

            combined_train_df = pd.concat([metadata_df_train, train_target_df], axis=1)

            combined_train_df.to_csv(train_csv_name, index=False)

            print(f"New best model saved (RMSE={val_rmse:.5f}) at epoch {epoch+1}")
            parity_plot(combined_val_df, parity_plot_path, "Validation Set: Epoch " + str(epoch+1),
            'val_true', 'val_pred')
        else:
            print(f"No improvement (best RMSE={best_val_rmse:.5f})")

        end_time = time.time()
        epoch_minutes = (end_time - start_time) / 60
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val RMSE: {val_rmse:.5f} | "
            f"Time: {epoch_minutes:.2f} min"
        )
        
        # For Plotting Learning Curve
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        plot_learning_curve(learning_curve_path, training_loss, validation_loss)

        learning_curve_df = pd.DataFrame({
            'training_loss': training_loss,
            'validation_loss': validation_loss
        })
        
        learning_curve_df.to_csv(learning_curve_df_path, index=False)
        
    
    ###### Prediction on Test Set:
    test_df = pd.read_csv(test_data_path)
    test_df, _, _, test_ids, _, _, T_scale_coef, P_scale_coef, log_P_scale_coef = train_val_test_split(
        df           = test_df,
        id_column    = structure_id_col,
        train_size   = 1.00,
        val_size     = 0.00,
        test_size    = 0.00, 
        random_state = 42,
        adsorption_model = ads_model
    )
    test_samples = prepare_data(
        test_df, h5_path,
        num_atomic_features, atomic_feature_scaling_coef,
        T_scale_coef, P_scale_coef, log_P_scale_coef,
        log_scaling_P=log_P, target_col = target_name, mode = mode_num
    )
    
    test_dataset = MOFDataset(test_samples)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=160,
        shuffle=False,
        num_workers=num_cpus,
        collate_fn=lambda batch: upsample_collate(batch, drop_points=False)
    )
    
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model.eval()
    
    ### Perform Prediction on Test Set + Calculate Statistics
    print("Evaluating on test set...")
    dummy_loss_fn = torch.nn.MSELoss()  
    test_loss, test_preds, test_targets, test_row_indicies = evaluate(model, test_loader, dummy_loss_fn, device)
    test_rmse = compute_rmse(test_targets, test_preds)
    test_r2 = compute_r2(test_targets, test_preds)
    
    
    valid_indicies = test_df.index.values
    valid_test_mask = np.isin(test_row_indicies, valid_indicies)
    test_safe_indices = test_row_indicies[valid_test_mask]
    test_metadata_df = test_df.loc[test_safe_indices].reset_index(drop=True)
    
    test_target_df = pd.DataFrame({
        'test_true': test_targets,
        'test_pred': test_preds
    })
    
    combined_test_df = pd.concat([test_metadata_df, test_target_df], axis=1)
    combined_test_df.to_csv(test_csv_name,index=False)
    print(f"Test predictions saved to {test_csv_name}")
    
    # Test Parity Plot
    parity_plot(combined_test_df, test_parity_plot_path, "Pointnet++ Regression (Test Prediction)", true_column='test_true', pred_column='test_pred')
    print(f"Parity plot saved to {test_parity_plot_path}")
