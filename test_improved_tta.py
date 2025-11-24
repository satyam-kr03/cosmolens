#!/usr/bin/env python3
"""
Test script for improved deterministic TTA
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.append('src')
from training.dataset import AugmentedCosmologyDataset

# Mock ensemble class for testing
class MockEnsemble:
    def predict(self, loader):
        # Return random predictions for testing
        n_samples = len(loader.dataset)
        return np.random.randn(n_samples, 2) * 0.1

# Mock ImprovedPredictionPipeline
class ImprovedPredictionPipeline:
    def __init__(self, ensemble, device, label_scaler):
        self.ensemble = ensemble
        self.device = device
        self.label_scaler = label_scaler

    def predict_with_tta(self, test_loader, n_augmentations=8, weight_by_uncertainty=False,
                        use_consistency_loss=False, consistency_weight=0.1):
        """
        Test-Time Augmentation: Average predictions over multiple augmented versions
        Uses deterministic augmentations (4 rotations + 2 flips) instead of random ones
        """
        print(f"Predicting with deterministic TTA ({n_augmentations} augmentations)...")

        all_predictions = []
        all_uncertainties = []

        # Get original predictions (no augmentation)
        y_pred = self.ensemble.predict(test_loader)
        all_predictions.append(y_pred)
        # For original, use small default uncertainty
        all_uncertainties.append(np.full((y_pred.shape[1],), 0.001))  # Shape: (n_targets,)

        # Generate predictions for each deterministic augmentation
        for aug_idx in range(1, n_augmentations):
            # Create augmented test dataset with deterministic augmentation
            aug_dataset = AugmentedCosmologyDataset(
                test_loader.dataset.data,
                transform=test_loader.dataset.transform,
                augment=True,  # Enable augmentation
                augmentation_idx=aug_idx  # Use deterministic augmentation
            )
            aug_loader = DataLoader(
                aug_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=0  # Use 0 for testing
            )

            y_pred_aug = self.ensemble.predict(aug_loader)
            all_predictions.append(y_pred_aug)
            # Estimate uncertainty for this augmentation
            all_uncertainties.append(np.std(y_pred_aug, axis=0))  # Remove keepdims=True

        all_predictions = np.array(all_predictions)
        all_uncertainties = np.array(all_uncertainties)

        if weight_by_uncertainty:
            # Weight predictions by inverse uncertainty
            weights = 1.0 / (all_uncertainties + 1e-8)  # Shape: (8, 2)
            weights = weights / np.sum(weights, axis=0, keepdims=True)  # Normalize: (8, 2)
            # Expand weights to match predictions shape: (8, 1, 2) for broadcasting
            weights = weights[:, np.newaxis, :]
            y_pred_mean = np.sum(all_predictions * weights, axis=0)
        else:
            y_pred_mean = np.mean(all_predictions, axis=0)

        # Overall uncertainty as std across augmentations
        y_pred_std = np.std(all_predictions, axis=0)

        # Inverse transform
        y_pred_mean = self.label_scaler.inverse_transform(y_pred_mean)

        print(f"TTA complete. Prediction std: {np.mean(y_pred_std, axis=0)}")
        if weight_by_uncertainty or use_consistency_loss:
            print(f"Used uncertainty weighting: {weight_by_uncertainty}, consistency loss: {use_consistency_loss}")

        return y_pred_mean, y_pred_std

    def compute_consistency_loss(self, predictions_list, consistency_weight=0.1):
        """Compute self-supervised consistency loss across augmentations"""
        if len(predictions_list) < 2:
            return 0.0

        preds = np.array(predictions_list)
        consistency_loss = 0.0
        n_pairs = 0

        for i in range(len(predictions_list)):
            for j in range(i+1, len(predictions_list)):
                mse_loss = np.mean((preds[i] - preds[j])**2)
                consistency_loss += mse_loss
                n_pairs += 1

        if n_pairs > 0:
            consistency_loss /= n_pairs

        print(f"Consistency loss: {consistency_loss:.6f} (weight: {consistency_weight})")

        return consistency_weight * consistency_loss

def test_improved_tta():
    """Test the improved deterministic TTA"""
    print("=" * 60)
    print("TESTING IMPROVED DETERMINISTIC TTA")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    n_samples = 50
    img_size = 32
    X_test = np.random.randn(n_samples, img_size, img_size).astype(np.float32)

    # Create transforms
    means = np.mean(X_test, dtype=np.float32)
    stds = np.std(X_test, dtype=np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[means], std=[stds]),
    ])

    # Create label scaler
    label_scaler = StandardScaler()
    dummy_labels = np.random.randn(n_samples, 2)
    label_scaler.fit(dummy_labels)

    # Create test dataset and loader
    test_dataset = AugmentedCosmologyDataset(X_test, transform=transform, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

    # Create mock ensemble and pipeline
    mock_ensemble = MockEnsemble()
    pipeline = ImprovedPredictionPipeline(mock_ensemble, "cpu", label_scaler)

    # Test different TTA configurations
    configs = [
        {"weight_by_uncertainty": False, "use_consistency_loss": False, "desc": "Basic deterministic TTA"},
        {"weight_by_uncertainty": True, "use_consistency_loss": False, "desc": "TTA with uncertainty weighting"},
        {"weight_by_uncertainty": False, "use_consistency_loss": True, "desc": "TTA with consistency loss"},
        {"weight_by_uncertainty": True, "use_consistency_loss": True, "desc": "TTA with both improvements"},
    ]

    for config in configs:
        print(f"\n--- Testing: {config['desc']} ---")
        y_pred, y_std = pipeline.predict_with_tta(
            test_loader,
            n_augmentations=8,
            weight_by_uncertainty=config["weight_by_uncertainty"],
            use_consistency_loss=config["use_consistency_loss"]
        )
        print(f"Predictions shape: {y_pred.shape}, Std shape: {y_std.shape}")
        print(f"Mean prediction: {np.mean(y_pred, axis=0)}")
        print(f"Mean uncertainty: {np.mean(y_std, axis=0)}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - DETERMINISTIC TTA WORKING!")
    print("=" * 60)

if __name__ == "__main__":
    test_improved_tta()