# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# | Based on: DeepSurv(czifan)                       |
# +--------------------------------------------------+

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
import numpy as np
import random
import time

from Model.model import DeepMCM
from Model.loss_function import DeepMCMLoss
from Model.dataset import SurvivalDataset
from Model.utils import read_config
from Model.utils import c_index
from Model.utils import adjust_learning_rate
from Model.utils import create_logger

import pandas as pd
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
def set_seed(seed):
    """Set the seed for random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load dataset based on configuration
def load_data(config, test_size, seed, save_dir=None, run_idx=None):
    """
    Load data from the specified CSV file, generate patient IDs, and split into training and test sets.

    Parameters:
        config: Dictionary containing configuration settings.
        test_size: Proportion of the dataset to include in the test split.
        seed: Random seed for reproducibility.
        save_dir: Directory to save the test set CSV file (optional).
        run_idx: Index of the current run for naming the test set file (optional).

    Returns:
        Training and test splits for covariates (X), survival times (y), and events (e).
    """
    data_file = config['train']['data_file']  # Path to the data file
    t_col = config['train']['t_col']          # Column for survival times
    e_col = config['train']['e_col']          # Column for event indicators
    x_cols = config['train']['x_cols']        # Columns for covariates

    # Load data from CSV
    data = pd.read_csv(data_file)

    # Generate PatientID column
    data['PatientID'] = np.arange(len(data))

    # Parse column settings
    t_col = t_col.strip()  # Survival time column
    e_col = e_col.strip()  # Event column

    # Process x_cols as a list of covariate columns
    if isinstance(x_cols, str):
        x_cols = [col.strip() for col in x_cols.split(',')]

    # Extract data for specified columns
    t = data[t_col].values.reshape(-1, 1)  # Survival times
    e = data[e_col].values.reshape(-1, 1)  # Event indicators
    e = (e == 1).astype(int)
    x = data[x_cols].values  # Covariates
    patient_ids = data['PatientID'].values  # PatientID column

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, e_train, e_test, id_train, id_test = train_test_split(
        x, t, e, patient_ids, test_size=test_size, random_state=seed)

    # Save the test set with PatientID to a CSV file if save_dir and run_idx are provided
    if save_dir is not None and run_idx is not None:
        test_df = pd.DataFrame(X_test, columns=x_cols)
        test_df['PatientID'] = id_test
        test_df['t'] = y_test
        test_df['e'] = e_test
        test_df.to_csv(os.path.join(save_dir, f'test_set_run_{run_idx}.csv'), index=False)

    return X_train, X_test, y_train, y_test, e_train, e_test


# Training function
def train(ini_file, models_dir, test_size=0.3, batch_size=64, patience=50, seed=None, cure_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], save_dir=None, run_idx=None):
    """
    Train the DeepMCM model using the specified configuration and parameters.

    Parameters:
        ini_file: Path to the configuration file.
        models_dir: Directory to save trained models.
        test_size: Proportion of data for testing.
        batch_size: Batch size for training.
        patience: Early stopping patience.
        seed: Random seed for reproducibility.
        cure_thresholds: List of cure thresholds to evaluate.
        save_dir: Directory to save the test set CSV file (optional).
        run_idx: Index of the current run for naming the test set file (optional).

    Returns:
        Best c-index and total training time.
    """
    if seed is not None:
        set_seed(seed)

    # Load configuration from ini file
    config = read_config(ini_file)

    # Ensure the models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Load data, passing save_dir and run_idx if provided
    X_train, X_test, y_train, y_test, e_train, e_test = load_data(config, test_size, seed, save_dir, run_idx)

    # Create datasets and data loaders
    train_dataset = SurvivalDataset(X_train, y_train, e_train)
    test_dataset = SurvivalDataset(X_test, y_test, e_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    best_threshold = cure_thresholds[0]
    best_c_index = 0
    best_epoch = 0
    best_model_state = None
    total_train_time = 0  # Initialize total training time
    flag = 0  # Counter for early stopping

    start_time = time.time()

    for cure_threshold in cure_thresholds:
        model = DeepMCM(config['pi_network'], config['surv_network'], device, cure_threshold=cure_threshold).to(device)
        criterion = DeepMCMLoss(config['regularization'], config['regularization']).to(device)
        optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
            model.parameters(), lr=config['train']['learning_rate'])

        for epoch in range(1, config['train']['epochs']+1):
            epoch_start_time = time.time()  # Record epoch start time
            lr = adjust_learning_rate(optimizer, epoch, config['train']['learning_rate'], config['train']['lr_decay_rate'])
            model.train()
            epoch_train_loss = 0
            all_risk_preds = []
            all_ys = []
            all_es = []
            for X, y, e in train_loader:
                X, y, e = X.to(device), y.to(device), e.to(device)
                pi_pred, risk_pred = model(X)
                train_loss = criterion(pi_pred, risk_pred, y, e, model.pi_network, model.surv_network)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_train_loss += train_loss.item() * X.size(0)
                all_risk_preds.append(risk_pred.detach().cpu().numpy())
                all_ys.append(y.cpu().numpy())
                all_es.append(e.cpu().numpy())
            epoch_train_loss /= len(train_dataset)
            all_risk_preds = np.concatenate(all_risk_preds)
            all_ys = np.concatenate(all_ys)
            all_es = np.concatenate(all_es)
            train_c = c_index(-all_risk_preds, all_ys, all_es)

            # Validation step
            model.eval()
            epoch_valid_loss = 0
            all_risk_preds = []
            all_ys = []
            all_es = []
            with torch.no_grad():
                for X, y, e in test_loader:
                    X, y, e = X.to(device), y.to(device), e.to(device)
                    pi_pred, risk_pred = model(X)
                    valid_loss = criterion(pi_pred, risk_pred, y, e, model.pi_network, model.surv_network)
                    epoch_valid_loss += valid_loss.item() * X.size(0)
                    all_risk_preds.append(risk_pred.detach().cpu().numpy())
                    all_ys.append(y.cpu().numpy())
                    all_es.append(e.cpu().numpy())
            epoch_valid_loss /= len(test_dataset)
            all_risk_preds = np.concatenate(all_risk_preds)
            all_ys = np.concatenate(all_ys)
            all_es = np.concatenate(all_es)
            valid_c = c_index(-all_risk_preds, all_ys, all_es)

            # Update the best model if current c-index is better
            if valid_c > best_c_index:
                best_c_index = valid_c
                best_threshold = cure_threshold
                best_epoch = epoch
                best_model_state = model.state_dict()  # Save model state

                # Save the model with the specified naming format
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, os.path.join(models_dir, os.path.basename(ini_file)+f'_seed{seed}.pth'))

            print(f'\rEpoch: {epoch}\tLoss: {epoch_train_loss:.8f}({epoch_valid_loss:.8f})\tc-index: {train_c:.8f}({valid_c:.8f})\tlr: {lr:g}', end='', flush=True)

            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time

    run_time = time.time() - start_time
    print(f'\nBest cure threshold: {best_threshold}, corresponding c-index: {best_c_index:.6f}, total time: {run_time:.2f} seconds')

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Label all patient data using the best model
    X_all = torch.tensor(np.concatenate([X_train, X_test]), dtype=torch.float32).to(device)
    pi_all = model.pi_network(X_all).cpu().detach().numpy()
    cured_labels = (pi_all >= best_threshold).astype(int)  # Cured as 1, not cured as 0

    # Save the results to a file
    result_df = pd.DataFrame({
        'PatientID': np.arange(len(cured_labels)),  # Assuming each patient has a unique ID
        'CuredLabel': cured_labels.flatten()
    })
    result_df.to_csv(os.path.join(cure_dir, f'{name}{best_c_index}_cured_labels.csv'), index=False)
    print(f"Cured labels saved to {name}{best_c_index}_cured_labels.csv")

    return best_c_index, run_time


# Main entry point
if __name__ == '__main__':
    # Global settings
    logs_dir = r'logs'
    output_dir = r'output'
    cure_dir = r'output\cure_label'
    models_dir = os.path.join(logs_dir, 'models')

    # Check and create logs_dir and models_dir
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Create directory to store c-index results
    cindex_dir = os.path.join(output_dir, 'cindex_results')
    if not os.path.exists(cindex_dir):
        os.makedirs(cindex_dir)

    # Create logger
    logger = create_logger(logs_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = r'configs'
    params = [
        ('pbc2', 'pbc2_filled.ini'),
        ('aids', 'aids.ini'),
        ('long_term', 'long_term_3x.ini'),
        ('short_term', 'short_term_3x.ini'),
        ('long_linear', 'long_linear.ini'),
        ('long_linear2', 'long_linear2.ini'),
        ('nonlin_data', 'nonlin_data.ini'),
    ]

    # Custom training parameters
    test_size = 0.3  # Proportion of the test set
    batch_size = 128  # Batch size for training
    patience = 80  # Early stopping patience
    num_runs = 10  # Number of training runs per configuration

    # Start training
    headers = []
    values = []
    time_headers = []
    time_values = []

    for name, ini_file in params:
        logger.info(f'Running {name} ({ini_file})...')
        c_indices = []
        times = []  # To track training times

        for run in range(1, num_runs + 1):
            logger.info(f'Run {run}/{num_runs} for {name}')
            seed = run  # Set seed for reproducibility

            # Call the train function, passing save_dir and run_idx
            best_c_index, train_time = train(
                os.path.join(configs_dir, ini_file),
                models_dir,
                test_size=test_size,          # Specify test_size during training
                batch_size=batch_size,        # Specify batch_size during training
                patience=patience,            # Specify patience during training
                seed=seed,
                run_idx=run                   # Pass the current run index
            )
            c_indices.append(best_c_index)
            times.append(train_time)
            logger.info(f'Run {run} completed with c-index: {best_c_index:.6f} in {train_time:.2f} seconds')

        # Calculate mean and standard deviation of c-index
        mean_c = np.mean(c_indices)
        std_c = np.std(c_indices)
        headers.append(name)
        values.append(f'{mean_c:.6f} ± {std_c:.6f}')
        
        # Calculate mean and standard deviation of training time
        mean_time = np.mean(times)
        std_time = np.std(times)
        time_headers.append(f'{name}_time')
        time_values.append(f'{mean_time:.2f} ± {std_time:.2f}')
        
        # Print and log training results
        logger.info(f"The c-index for {name}: {mean_c:.6f} ± {std_c:.6f}")
        logger.info(f"The training time for {name}: {mean_time:.2f} ± {std_time:.2f} seconds")

        # Write c-index results to a text file
        cindex_file = os.path.join(cindex_dir, f'cindex_{name}.txt')
        with open(cindex_file, 'w', encoding='utf-8') as f:
            for run_idx, c in enumerate(c_indices, 1):
                f.write(f'Run {run_idx}: {c:.6f}\n')
        logger.info(f'c-index results saved to {cindex_file}\n')

    # Print results table
    tb = pt.PrettyTable()
    tb.field_names = headers + time_headers
    tb.add_row(values + time_values)
    logger.info(tb)
