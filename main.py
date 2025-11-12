import os
import json
import time
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import optuna

# Import our modular components
from src.utils import Utility, Data, Score
from src.models import MultiScaleResNetWithHead, StackedEnsemble
from src.features import compute_statistical_features, HybridFeatureExtractor
from src.training import AugmentedCosmologyDataset, train_epoch, validate_epoch, Config
from src.inference import ImprovedPredictionPipeline


def main():
    """Main execution function"""
    print("Cosmology Parameter Estimation Pipeline")
    print("=" * 50)

    # Configuration
    root_dir = os.getcwd()
    USE_PUBLIC_DATASET = True
    PUBLIC_DATA_DIR = './data'
    DATA_DIR = PUBLIC_DATA_DIR if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')

    config = Config()
    print(f"Device: {config.DEVICE}")

    # Load data
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.load_train_data()
    data_obj.load_test_data()

    print(f"Train shape: {data_obj.kappa.shape}, Test shape: {data_obj.kappa_test.shape}")

    # Data preprocessing
    noisy_kappa_train = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_train.npy")
    label_train = Utility.load_np(data_dir=DATA_DIR, file_name="label_train.npy")
    noisy_kappa_val = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_val.npy")
    label_val = Utility.load_np(data_dir=DATA_DIR, file_name="label_val.npy")

    Ntrain = label_train.shape[0] * label_train.shape[1]
    Nval = label_val.shape[0] * label_val.shape[1]

    X_train = noisy_kappa_train.reshape(Ntrain, *data_obj.shape)
    X_val = noisy_kappa_val.reshape(Nval, *data_obj.shape)
    y_train = label_train.reshape(Ntrain, 5)[:, :2]
    y_val = label_val.reshape(Nval, 5)[:, :2]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Data normalization
    means = np.mean(X_train, dtype=np.float32)
    stds = np.std(X_train, dtype=np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[means], std=[stds]),
    ])

    label_scaler = StandardScaler()
    y_train_scaled = label_scaler.fit_transform(y_train)
    y_val_scaled = label_scaler.transform(y_val)

    train_dataset = AugmentedCosmologyDataset(X_train, y_train_scaled, transform, augment=True)
    val_dataset = AugmentedCosmologyDataset(X_val, y_val_scaled, transform, augment=False)
    test_dataset = AugmentedCosmologyDataset(data_obj.kappa_test, transform=transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Loaders ready: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")

    # Initialize model
    config.IMG_HEIGHT = data_obj.shape[0]
    config.IMG_WIDTH = data_obj.shape[1]

    cnn_model = MultiScaleResNetWithHead(
        height=config.IMG_HEIGHT,
        width=config.IMG_WIDTH,
        num_targets=config.NUM_TARGETS,
        feature_dim=config.FEATURE_DIM,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    print(f"Multi-scale ResNet initialized with {sum(p.numel() for p in cnn_model.parameters())} parameters")


    # Hyperparameter optimization with Optuna
    USE_OPTUNA = False
    N_TRIALS = 5

    if USE_OPTUNA:
        print("Starting Optuna hyperparameter optimization...")

        def objective(trial):
            # Suggest hyperparameters
            feature_dim = trial.suggest_categorical('feature_dim', [128, 256, 512])
            dropout = trial.suggest_float('dropout', 0.2, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            # XGBoost params
            xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 100, 500)
            xgb_max_depth = trial.suggest_int('xgb_max_depth', 5, 15)
            xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.1)
            xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 0.9)
            xgb_colsample = trial.suggest_float('xgb_colsample', 0.6, 0.9)
            xgb_reg_alpha = trial.suggest_float('xgb_reg_alpha', 0.01, 1.0)
            xgb_reg_lambda = trial.suggest_float('xgb_reg_lambda', 0.1, 2.0)

            pca_components = trial.suggest_categorical('pca_components', [64, 128, 256])

            # Create dataloaders with trial batch size
            trial_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            trial_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize model
            trial_model = MultiScaleResNetWithHead(
                height=data_obj.shape[0],
                width=data_obj.shape[1],
                num_targets=config.NUM_TARGETS,
                feature_dim=feature_dim,
                dropout=dropout
            ).to(config.DEVICE)

            # Train CNN for fewer epochs
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.AdamW(trial_model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1)

            n_epochs = 5  # Reduced for faster optimization
            for epoch in range(n_epochs):
                train_loss = train_epoch(trial_model, trial_train_loader, loss_fn, optimizer, config.DEVICE)
                val_loss = validate_epoch(trial_model, trial_val_loader, loss_fn, config.DEVICE)
                scheduler.step()

                # Report intermediate value for pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Train ensemble with trial XGBoost params
            trial_xgb_params = {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'subsample': xgb_subsample,
                'colsample_bytree': xgb_colsample,
                'reg_alpha': xgb_reg_alpha,
                'reg_lambda': xgb_reg_lambda,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }

            trial_ensemble = StackedEnsemble(trial_model, config.DEVICE,
                                             xgb_params=trial_xgb_params,
                                             pca_components=pca_components)
            trial_ensemble.fit(trial_train_loader, num_targets=config.NUM_TARGETS)

            # Evaluate on validation set
            y_pred_trial = trial_ensemble.predict(trial_val_loader)
            y_pred_trial = label_scaler.inverse_transform(y_pred_trial)

            # Compute MSE as objective
            mse = mean_squared_error(y_val, y_pred_trial)

            return mse

        # Create study with pruning
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        print("\n" + "=" * 70)
        print("OPTUNA OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Best MSE: {study.best_value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 70)

        # Update config with best parameters
        config.FEATURE_DIM = study.best_params['feature_dim']
        config.DROPOUT = study.best_params['dropout']
        config.LEARNING_RATE = study.best_params['learning_rate']
        config.BATCH_SIZE = study.best_params['batch_size']
        config.WEIGHT_DECAY = study.best_params['weight_decay']
        config.PCA_COMPONENTS = study.best_params['pca_components']

        config.XGB_PARAMS = {
            'n_estimators': study.best_params['xgb_n_estimators'],
            'max_depth': study.best_params['xgb_max_depth'],
            'learning_rate': study.best_params['xgb_learning_rate'],
            'subsample': study.best_params['xgb_subsample'],
            'colsample_bytree': study.best_params['xgb_colsample'],
            'reg_alpha': study.best_params['xgb_reg_alpha'],
            'reg_lambda': study.best_params['xgb_reg_lambda'],
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        # Recreate dataloaders with optimized batch size
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

        print("\nConfig updated with best hyperparameters!")

    else:
        print("Skipping Optuna optimization (USE_OPTUNA=False)")

    # Train CNN feature extractor
    USE_PRETRAINED = False

    if not USE_PRETRAINED:
        print("Training CNN feature extractor...")

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(cnn_model, train_loader, loss_fn, optimizer, config.DEVICE)
            val_loss = validate_epoch(cnn_model, val_loader, loss_fn, config.DEVICE)
            scheduler.step()

            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(cnn_model.state_dict(), config.CNN_MODEL_PATH)
                print("  ✓ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        cnn_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, weights_only=True))
        print("CNN training complete!")

    else:
        cnn_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, weights_only=True))
        print("Loaded pretrained CNN")

    # Train stacked ensemble
    if not USE_PRETRAINED:
        print("Training stacked ensemble...")

        feature_extractor = cnn_model.get_feature_extractor()
        ensemble = StackedEnsemble(cnn_model, config.DEVICE, xgb_params=config.XGB_PARAMS,
                                   pca_components=config.PCA_COMPONENTS)
        ensemble.fit(train_loader, num_targets=config.NUM_TARGETS)

        with open(config.ENSEMBLE_PATH, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"Ensemble saved to {config.ENSEMBLE_PATH}")

    else:
        with open(config.ENSEMBLE_PATH, 'rb') as f:
            ensemble = pickle.load(f)
        print("Loaded pretrained ensemble")

    # Validation predictions and MCMC
    print("Predicting on validation set...")
    y_pred_val = ensemble.predict(val_loader)
    y_pred_val = label_scaler.inverse_transform(y_pred_val)

    print(f"Predictions: {y_pred_val.shape}")

    # MCMC setup
    cosmology = data_obj.label[:,0,:2]
    Ncosmo = data_obj.Ncosmo

    row_to_i = {tuple(cosmology[i]): i for i in range(Ncosmo)}
    index_lists = [[] for _ in range(Ncosmo)]

    for idx in range(len(y_val)):
        row_tuple = tuple(y_val[idx])
        i = row_to_i[row_tuple]
        index_lists[i].append(idx)

    val_cosmology_idx = [np.array(lst) for lst in index_lists]

    d_vector = []
    n_d = 2

    for i in range(Ncosmo):
        d_i = np.zeros((len(val_cosmology_idx[i]), n_d))
        for j, idx in enumerate(val_cosmology_idx[i]):
            d_i[j] = y_pred_val[idx]
        d_vector.append(d_i)

    mean_d_vector = np.array([np.mean(d_vector[i], 0) for i in range(Ncosmo)])
    delta = [d_vector[i] - mean_d_vector[i].reshape(1, n_d) for i in range(Ncosmo)]
    cov_d_vector = np.concatenate([(delta[i].T @ delta[i] / (len(delta[i])-n_d-2))[None]
                                    for i in range(Ncosmo)], 0)

    mean_d_vector_interp = LinearNDInterpolator(cosmology, mean_d_vector, fill_value=np.nan)
    cov_d_vector_interp = LinearNDInterpolator(cosmology, cov_d_vector, fill_value=np.nan)
    logprior_interp = LinearNDInterpolator(cosmology, np.zeros((Ncosmo, 1)), fill_value=-np.inf)

    def log_prior(x):
        return logprior_interp(x).flatten()

    def loglike(x, d):
        mean = mean_d_vector_interp(x)
        cov = cov_d_vector_interp(x)
        delta = d - mean
        inv_cov = np.linalg.inv(cov)
        cov_det = np.linalg.slogdet(cov)[1]
        return -0.5 * cov_det - 0.5 * np.einsum("ni,nij,nj->n", delta, inv_cov, delta)

    def logp_posterior(x, d):
        logp = log_prior(x)
        select = np.isfinite(logp)
        if np.sum(select) > 0:
            logp[select] = logp[select] + loglike(x[select], d[select])
        return logp

    print("MCMC setup complete")

    # Run MCMC on validation set
    Nstep = 10000
    sigma = 0.06

    current = cosmology[np.random.choice(Ncosmo, size=Nval)]
    curr_logprob = logp_posterior(current, y_pred_val)

    states = []
    total_acc = np.zeros(len(current))

    print("Running MCMC on validation set...")

    for i in tqdm(range(Nstep), desc="MCMC"):
        proposal = current + np.random.randn(*current.shape) * sigma
        proposal_logprob = logp_posterior(proposal, y_pred_val)

        acc_logprob = proposal_logprob - curr_logprob
        acc_logprob[acc_logprob > 0] = 0
        acc_prob = np.exp(acc_logprob)
        acc = np.random.uniform(size=len(current)) < acc_prob

        total_acc += acc_prob
        current[acc] = proposal[acc]
        curr_logprob[acc] = proposal_logprob[acc]
        states.append(np.copy(current)[None])

    states = np.concatenate(states[int(0.2*Nstep):], 0)
    mean_val = np.mean(states, 0)
    errorbar_val = np.std(states, 0)

    print(f"MCMC complete! Acceptance rate: {np.mean(total_acc/Nstep):.3f}")
    print(f"Mean error bars: {np.mean(errorbar_val, 0)}")

    # Validation score
    validation_score = Score._score_phase1(y_val, mean_val, errorbar_val)
    print(f"\nValidation Score: {validation_score:.2f}")
    print(f"Error bar (Ωₘ): {np.mean(errorbar_val[:, 0]):.6f}")
    print(f"Error bar (S₈): {np.mean(errorbar_val[:, 1]):.6f}")

    # Test predictions with improved pipeline
    print("=" * 70)
    print("GENERATING IMPROVED TEST PREDICTIONS")
    print("=" * 70)

    # Initialize improved pipeline
    improved_pipeline = ImprovedPredictionPipeline(ensemble, config.DEVICE, label_scaler)

    # Test-Time Augmentation on test set
    y_pred_test_tta, y_pred_test_std = improved_pipeline.predict_with_tta(
        test_loader,
        n_augmentations=8  # More augmentations for test set
    )

    print(f"\nTest predictions with TTA complete: {y_pred_test_tta.shape}")

    # Run multiple MCMC chains on test set
    states_test_improved = improved_pipeline.run_multiple_mcmc_chains(
        y_pred_test_tta,
        cosmology,
        mean_d_vector_interp,
        cov_d_vector_interp,
        logprior_interp,
        n_chains=4,  # More chains for test set
        n_steps=12000,  # More steps for better convergence
        sigma=0.05,
        burn_in=0.25
    )

    mean_test_improved = np.mean(states_test_improved, 0)
    errorbar_test_improved = np.std(states_test_improved, 0)

    print(f"\nTest MCMC complete!")
    print(f"Mean error bars: {np.mean(errorbar_test_improved, 0)}")

    # Create improved submission
    data_submission_improved = {
        "means": mean_test_improved.tolist(),
        "errorbars": errorbar_test_improved.tolist()
    }

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = f'Submission_ImprovedPipeline_{timestamp}.zip'

    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data_submission_improved
    )

    print("\n" + "=" * 70)
    print("IMPROVED SUBMISSION CREATED")
    print("=" * 70)
    print(f"File: {zip_file}")
    print(f"Test samples: {len(mean_test_improved)}")
    print("Improvements applied:")
    print("  ✓ Test-Time Augmentation (8 augmentations)")
    print("  ✓ Multiple MCMC chains (4 chains, 12000 steps)")
    print("  ✓ Adaptive step size")
    print("  ✓ Calibrated error bars")
    print("=" * 70)

    print("Pipeline setup complete!")


if __name__ == "__main__":
    main()