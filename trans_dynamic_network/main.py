# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: Junbei Zhang                             |
# | Based on: Dynamic-DeepHit                        |
# +--------------------------------------------------+

import numpy as np  # Import numpy for array operations
import pandas as pd  # Import pandas for data processing
import os  # Import os for interacting with the file system
import torch  # Import PyTorch for building models

from sklearn.model_selection import train_test_split  # Import function for data splitting
from tqdm import tqdm  # Import tqdm for progress bar display
import import_data as impt  # Import custom data import module
from Model import Dynamic2StaticNet  # Import the new Dynamic2StaticNet model
import time as TIME  # Import time for performance tracking
from torch.utils.data import DataLoader  # Import DataLoader for batch processing

# Set dataset and configuration
data_mode = 'PBC2'      # If you are using PBC2, set it to PBC2. If you want to use another dataset, modify the data path and column declarations in the 'import_dataset_other' function within import_data.py;
                        #Also, replace the corresponding dataset name here (including the second data_mode below) to match the name specified here.
seed = 1234

# Load dataset
if data_mode == 'PBC2':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), data_mi = impt.import_dataset(
        norm_mode='standard')
elif data_mode == 'aids':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), data_mi = impt.import_dataset_other(
        norm_mode='standard')
else:
    print('Error: Specified data mode not found!')

_, num_Event, num_Category = np.shape(mask1)  # Dimensions of mask1: [number of samples, number of events, number of categories]
max_length = np.shape(data)[1]

# Set hyperparameters (focus mainly on feature extraction, training settings can be ignored)
new_parser = {
    'mb_size': 32,
    'keep_prob': 0.6,
    'h_dim_Transformer': 100,
    'h_dim_FC': 100,
    'num_layers_Transformer': 2,
    'FC_active_fn': 'ReLU',
}

# Dynamically calculate nhead to ensure input_size is divisible by nhead
def get_appropriate_nhead(input_size, max_nhead=8):
    for n in range(min(input_size, max_nhead), 0, -1):
        if input_size % n == 0:
            return n
    return 1  # If no appropriate nhead is found, set to 1

input_size_for_transformer = x_dim * 2  # Because TransformerModel's input_size is input_dim * 2
nhead = get_appropriate_nhead(input_size_for_transformer)
print(f"Selected nhead: {nhead}, based on input_size: {input_size_for_transformer}")

# Add nhead to new_parser
new_parser['nhead'] = nhead

# Define input dimensions and network settings
network_settings = {
    'h_dim_Transformer': new_parser['h_dim_Transformer'],
    'h_dim_FC': new_parser['h_dim_FC'],
    'num_layers_Transformer': new_parser['num_layers_Transformer'],
    'FC_active_fn': new_parser['FC_active_fn'],
    'keep_prob': new_parser['keep_prob'],
    'num_Event': num_Event,
    'num_Category': num_Category,
    'nhead': new_parser['nhead'],  # Add nhead
}

# Set device (use GPU if available, otherwise use CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# If multiple GPUs are available, specify the GPU number to use
if torch.cuda.is_available():
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

# Create DataLoader for training data
train_data = impt.Dataset_all(data, data_mi, time, label, mask1, mask2, mask3)
train_data_loader = DataLoader(train_data, batch_size=new_parser['mb_size'], shuffle=False)

# Feature extraction function
def extract_features(train_data_loader):
    # Initialize Dynamic2StaticNet model and move it to the specified device
    model = Dynamic2StaticNet(
        input_dim=x_dim,
        output_dim=num_Category,
        network_settings=network_settings,
        risks=num_Event
    ).to(device)
    model.eval()  # Set model to evaluation mode since we only need feature extraction

    static_features = []

    # Start recording feature extraction time
    start_time = TIME.time()

    # Iterate over batches in the data loader
    with torch.no_grad():  # Disable gradient computation
        for data_ in tqdm(train_data_loader):
            data_batch = data_[0].to(device)  # Dynamic features
            data_mi_batch = data_[1].to(device)  # Missing data indicators (if applicable)

            # Extract static features through the model's forward pass
            static_feature_batch = model(data_batch, data_mi_batch)

            # Append the extracted static features to the list
            static_features.append(static_feature_batch.cpu().numpy())

    # End time recording
    end_time = TIME.time()

    # Calculate and print total time taken
    total_time = end_time - start_time
    print(f"Feature extraction completed, total time: {total_time:.2f} seconds")

    # Convert the list of static features to a numpy array for further processing
    static_features = np.concatenate(static_features, axis=0)
    return static_features

# Save extracted static features
def save_static_features(features, file_name='static_features.npy'):
    np.save(file_name, features)  # Save static features as a numpy array
    print(f"Static features saved to {file_name}")

# Function to get static survival data
def get_static_survival_data(static_features):
    # Extract id, tte (time to event), and label columns from the original data
    survival_data = pd.DataFrame({
        'id': np.arange(len(time)),
        'tte': time.flatten(),
        'label': label.flatten()
    })

    # Remove duplicates based on 'id' and keep the first occurrence
    survival_data_unique = survival_data.drop_duplicates(subset='id').sort_values(by='id').reset_index(drop=True)

    # Merge static features with survival data
    static_features_df = pd.DataFrame(static_features, index=survival_data_unique['id'])
    static_survival_data = pd.concat([survival_data_unique, static_features_df.reset_index(drop=True)], axis=1)

    return static_survival_data

# Save static survival data as CSV
def save_static_survival_data(static_survival_data, file_name='static.csv'):
    static_survival_data.to_csv(file_name, index=False)
    print(f"Static survival data saved to {file_name}")

# Save the dynamic model after feature extraction
def save_dynamic_model(model, file_path='dynamic_model.pth'):
    torch.save(model.state_dict(), file_path)
    print(f"Dynamic model saved to {file_path}")

# Main function to perform feature extraction and process survival data
if __name__ == "__main__":
    # Initialize Dynamic2StaticNet model and load it to the device
    model = Dynamic2StaticNet(
        input_dim=x_dim,
        output_dim=num_Category,
        network_settings=network_settings,
        risks=num_Event
    ).to(device)
    model.eval()  # Set to evaluation mode since it's only used for feature extraction

    # Create DataLoader for training data
    train_data = impt.Dataset_all(data, data_mi, time, label, mask1, mask2, mask3)
    train_data_loader = DataLoader(train_data, batch_size=new_parser['mb_size'], shuffle=False)

    # Extract static features
    static_features = extract_features(train_data_loader)

    # Generate the filename for static survival data based on the data mode
    static_survival_data_file = os.path.join(r'static_data', f"{data_mode}_static.csv")

    # Merge static features with survival information to create static survival data
    static_survival_data = get_static_survival_data(static_features)

    # Save static survival data as a CSV file for later use
    save_static_survival_data(static_survival_data, file_name=static_survival_data_file)

    # Optionally save the dynamic model
    # save_dynamic_model(model, os.path.join('model_trained', f'{data_mode}_dynamic_model.pth'))
