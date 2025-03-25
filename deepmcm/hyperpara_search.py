# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# +--------------------------------------------------+

import optuna
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from Model.model import DeepMCM
from Model.loss_function import DeepMCMLoss
from Model.dataset import SurvivalDataset
from Model.utils import read_config, c_index, adjust_learning_rate
from sklearn.model_selection import train_test_split

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load data function
def load_data(config, test_size, seed):
    """
    Load data based on configuration and split into training and test sets.
    """
    data_file = config['train']['data_file']
    t_col = config['train']['t_col']
    e_col = config['train']['e_col']
    x_cols = config['train']['x_cols']

    data = pd.read_csv(data_file)

    # Process column names
    t_col = t_col.strip()
    e_col = e_col.strip()
    if isinstance(x_cols, str):
        x_cols = [col.strip() for col in x_cols.split(',')]

    t = data[t_col].values.reshape(-1, 1)
    e = data[e_col].values.reshape(-1, 1)
    e = (e == 1).astype(int)
    x = data[x_cols].values

    # Split data
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        x, t, e, test_size=test_size, random_state=1234
    )
    return X_train, X_test, y_train, y_test, e_train, e_test

# Dynamically generate network layer configurations
def generate_dims(trial, input_dim, min_layers=2, max_layers=3, min_units=8, max_units=64):
    """
    Dynamically generate the number of layers and the number of neurons per layer for the network,
    ensuring that the input dimensions match.
    """
    num_layers = trial.suggest_int("num_layers", min_layers, max_layers)
    dims = [input_dim]  # Ensure the input dimension of the first layer matches the number of input features
    dims.extend([
        trial.suggest_int(f"hidden_units_layer_{i+1}", min_units, max_units, step=8)
        for i in range(num_layers - 1)
    ])
    dims.append(1)  # Ensure the output of the last layer is 1
    return dims

def objective(trial, config, device, test_size=0.4, seed=42):
    set_seed(seed)

    try:
        # Load data
        X_train, X_test, y_train, y_test, e_train, e_test = load_data(config, test_size, seed)
        input_dim = X_train.shape[1]  # Get the number of input features

        # Search global parameters
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
        lr_decay_rate = trial.suggest_float("lr_decay_rate", 1e-3, 1e-1, log=True)

        # Search parameters for pi_network and surv_network
        pi_drop = trial.suggest_float("pi_drop", 0.0, 0.7)
        pi_norm = trial.suggest_categorical("pi_norm", [True, False])
        pi_dims = generate_dims(trial, input_dim)

        surv_drop = trial.suggest_float("surv_drop", 0.0, 0.7)
        surv_norm = trial.suggest_categorical("surv_norm", [True, False])
        surv_dims = generate_dims(trial, input_dim)

        pi_activation = trial.suggest_categorical("pi_activation", ["ReLU", "SELU"])
        surv_activation = trial.suggest_categorical("surv_activation", ["ReLU", "SELU"])

        # Convert to config dictionaries
        pi_config = {
            "dims": pi_dims,
            "drop": pi_drop,
            "norm": pi_norm,
            "activation": pi_activation
        }
        surv_config = {
            "dims": surv_dims,
            "drop": surv_drop,
            "norm": surv_norm,
            "activation": surv_activation
        }

        # Data loaders
        train_dataset = SurvivalDataset(X_train, y_train, e_train)
        test_dataset = SurvivalDataset(X_test, y_test, e_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

        # Model and loss function
        model = DeepMCM(pi_config, surv_config, device, cure_threshold=0.7).to(device)
        criterion = DeepMCMLoss(config['regularization'], config['regularization']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training and validation loop
        best_c_index = 0
        for epoch in range(1, config['train']['epochs'] + 1):
            model.train()
            for X, y, e in train_loader:
                X, y, e = X.to(device), y.to(device), e.to(device)
                pi_pred, risk_pred = model(X)
                
                # Check the validity of outputs
                if torch.isnan(pi_pred).any() or torch.isnan(risk_pred).any():
                    raise ValueError("Model output contains NaN values.")
                
                loss = criterion(pi_pred, risk_pred, y, e, model.pi_network, model.surv_network)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            all_risk_preds, all_ys, all_es = [], [], []
            with torch.no_grad():
                for X, y, e in test_loader:
                    X, y, e = X.to(device), y.to(device), e.to(device)
                    _, risk_pred = model(X)
                    all_risk_preds.append(risk_pred.cpu().numpy())
                    all_ys.append(y.cpu().numpy())
                    all_es.append(e.cpu().numpy())
            all_risk_preds = np.concatenate(all_risk_preds)
            all_ys = np.concatenate(all_ys)
            all_es = np.concatenate(all_es)

            # Check for NaN
            if np.isnan(all_risk_preds).any() or np.isnan(all_ys).any() or np.isnan(all_es).any():
                raise ValueError("NaN detected in validation predictions or labels.")

            valid_c = c_index(-all_risk_preds, all_ys, all_es)

        return -valid_c

    except ValueError as e:
        print(f"Trial {trial.number} failed due to error: {e}")
        raise optuna.exceptions.TrialPruned()

# Main program
def run_optuna(config_file, device, n_trials=50):
    # Read configuration file
    config = read_config(config_file)

    # Start Optuna hyperparameter search
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, config=config, device=device), n_trials=n_trials)

    # Output the best results
    best_params = study.best_params
    best_c_index = -study.best_value  # Convert back to positive value
    print("Best parameters:", best_params)
    print("Best C-index:", best_c_index)

    # Visualization of the optimization process (optional)
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).show()
        vis.plot_param_importances(study).show()
    except ImportError:
        print("optuna.visualization is not installed, skipping visualization.")

    return best_params, best_c_index

# Configuration and running
if __name__ == "__main__":
    config_file = r"deepmcm\configs\pbc2_filled.ini"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params, best_c_index = run_optuna(config_file, device, n_trials=50)
