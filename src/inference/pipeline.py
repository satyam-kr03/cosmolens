import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

from ..training.dataset import AugmentedCosmologyDataset
from ..utils.utility import Utility
from ..utils.score import Score


class ImprovedPredictionPipeline:
    def __init__(self, ensemble, device, label_scaler):
        self.ensemble = ensemble
        self.device = device
        self.label_scaler = label_scaler

    def predict_with_tta(self, test_loader, n_augmentations=8):
        """
        Test-Time Augmentation: Average predictions over multiple augmented versions
        """
        print(f"Predicting with TTA ({n_augmentations} augmentations)...")

        all_predictions = []

        # Get original predictions
        y_pred = self.ensemble.predict(test_loader)
        all_predictions.append(y_pred)

        # Generate augmented predictions
        for aug_idx in range(n_augmentations - 1):
            # Create augmented test dataset
            aug_dataset = AugmentedCosmologyDataset(
                test_loader.dataset.data,
                transform=test_loader.dataset.transform,
                augment=True  # Enable augmentation
            )
            aug_loader = torch.utils.data.DataLoader(
                aug_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=4
            )

            y_pred_aug = self.ensemble.predict(aug_loader)
            all_predictions.append(y_pred_aug)

        # Average all predictions
        y_pred_mean = np.mean(all_predictions, axis=0)
        y_pred_std = np.std(all_predictions, axis=0)

        # Inverse transform
        y_pred_mean = self.label_scaler.inverse_transform(y_pred_mean)

        print(f"TTA complete. Prediction std: {np.mean(y_pred_std, axis=0)}")

        return y_pred_mean, y_pred_std

    def create_rbf_interpolators(self, cosmology, mean_d_vector, cov_d_vector):
        """
        Create RBF interpolators for smoother predictions
        """
        print("Creating RBF interpolators...")

        # Use RBF interpolation for smoother results
        mean_interp = []
        cov_interp = []

        for i in range(mean_d_vector.shape[1]):
            mean_interp.append(
                RBFInterpolator(cosmology, mean_d_vector[:, i:i+1],
                              kernel='thin_plate_spline', smoothing=0.01)
            )

        for i in range(cov_d_vector.shape[1]):
            for j in range(cov_d_vector.shape[2]):
                cov_interp.append(
                    RBFInterpolator(cosmology, cov_d_vector[:, i, j:j+1],
                                  kernel='thin_plate_spline', smoothing=0.01)
                )

        return mean_interp, cov_interp

    def logp_posterior_rbf(self, x, d, mean_interp, cov_interp, logprior_interp):
        """
        Posterior probability using RBF interpolation
        """
        logp = logprior_interp(x).flatten()
        select = np.isfinite(logp)

        if np.sum(select) > 0:
            # Get mean prediction
            mean = np.column_stack([interp(x[select]) for interp in mean_interp])

            # Get covariance
            n_d = len(mean_interp)
            cov = np.zeros((np.sum(select), n_d, n_d))
            idx = 0
            for i in range(n_d):
                for j in range(n_d):
                    cov[:, i, j] = cov_interp[idx](x[select]).flatten()
                    idx += 1

            # Compute log-likelihood
            delta = d[select] - mean
            inv_cov = np.linalg.inv(cov)
            cov_det = np.linalg.slogdet(cov)[1]
            loglike = -0.5 * cov_det - 0.5 * np.einsum("ni,nij,nj->n", delta, inv_cov, delta)

            logp[select] = logp[select] + loglike

        return logp

    def run_multiple_mcmc_chains(self, y_pred, cosmology, mean_d_vector_interp,
                                  cov_d_vector_interp, logprior_interp,
                                  n_chains=4, n_steps=10000, sigma=0.06, burn_in=0.2):
        """
        Run multiple MCMC chains and combine results
        """
        print(f"Running {n_chains} MCMC chains with {n_steps} steps each...")

        all_states = []
        acceptance_rates = []

        for chain_idx in range(n_chains):
            print(f"\nChain {chain_idx + 1}/{n_chains}")

            # Initialize from different starting points
            current = cosmology[np.random.choice(len(cosmology), size=len(y_pred))]

            # Compute posterior
            def logp_posterior(x, d):
                logp = logprior_interp(x).flatten()
                select = np.isfinite(logp)
                if np.sum(select) > 0:
                    mean = mean_d_vector_interp(x[select])
                    cov = cov_d_vector_interp(x[select])
                    delta = d[select] - mean
                    inv_cov = np.linalg.inv(cov)
                    cov_det = np.linalg.slogdet(cov)[1]
                    logp[select] = logp[select] - 0.5 * cov_det - 0.5 * np.einsum("ni,nij,nj->n", delta, inv_cov, delta)
                return logp

            curr_logprob = logp_posterior(current, y_pred)

            states = []
            total_acc = np.zeros(len(current))

            # Adaptive step size
            current_sigma = sigma

            for i in tqdm(range(n_steps), desc=f"MCMC Chain {chain_idx + 1}"):
                proposal = current + np.random.randn(*current.shape) * current_sigma
                proposal_logprob = logp_posterior(proposal, y_pred)

                acc_logprob = proposal_logprob - curr_logprob
                acc_logprob[acc_logprob > 0] = 0
                acc_prob = np.exp(acc_logprob)
                acc = np.random.uniform(size=len(current)) < acc_prob

                total_acc += acc_prob
                current[acc] = proposal[acc]
                curr_logprob[acc] = proposal_logprob[acc]
                states.append(np.copy(current)[None])

                # Adapt step size every 500 steps
                if (i + 1) % 500 == 0:
                    acc_rate = np.mean(total_acc / (i + 1))
                    if acc_rate < 0.2:
                        current_sigma *= 0.9
                    elif acc_rate > 0.4:
                        current_sigma *= 1.1

            # Remove burn-in
            states = np.concatenate(states[int(burn_in * n_steps):], 0)
            all_states.append(states)

            acceptance_rate = np.mean(total_acc / n_steps)
            acceptance_rates.append(acceptance_rate)
            print(f"Chain {chain_idx + 1} acceptance rate: {acceptance_rate:.3f}")

        # Combine all chains
        combined_states = np.concatenate(all_states, axis=0)

        print(f"\nCombined {n_chains} chains: {combined_states.shape[0]} total samples")
        print(f"Mean acceptance rate: {np.mean(acceptance_rates):.3f}")

        return combined_states

    def calibrate_error_bars(self, val_predictions, val_true, val_error_bars):
        """
        Calibrate error bars based on validation performance
        """
        print("Calibrating error bars...")

        # Compute actual errors
        actual_errors = np.abs(val_predictions - val_true)

        # Compute calibration factors for each parameter
        calibration_factors = []
        for i in range(val_true.shape[1]):
            # Ratio of actual error to predicted error
            ratio = actual_errors[:, i] / (val_error_bars[:, i] + 1e-8)
            # Use median for robustness
            calibration_factor = np.median(ratio)
            calibration_factors.append(calibration_factor)

        calibration_factors = np.array(calibration_factors)

        print(f"Calibration factors: {calibration_factors}")

        return calibration_factors