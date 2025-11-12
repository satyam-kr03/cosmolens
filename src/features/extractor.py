import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_statistical_features(images):
    """
    Compute statistical features from images.
    Args:
        images: numpy array of shape (N, H, W)
    Returns:
        features: numpy array of shape (N, n_features)
    """
    N = images.shape[0]
    features_list = []

    for i in range(N):
        img = images[i]
        img_flat = img.flatten()
        img_flat = img_flat[img_flat != 0]  # Remove masked values

        if len(img_flat) == 0:
            features_list.append(np.zeros(8))
            continue

        feat = [
            np.mean(img_flat),
            np.std(img_flat),
            skew(img_flat),
            kurtosis(img_flat),
            np.percentile(img_flat, 25),
            np.percentile(img_flat, 75),
            np.min(img_flat),
            np.max(img_flat)
        ]
        features_list.append(feat)

    return np.array(features_list, dtype=np.float32)


class HybridFeatureExtractor:
    def __init__(self, feature_extractor, device, use_pca=True, pca_components=128):
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self.stat_scaler = StandardScaler()

    def extract_features(self, dataloader, fit_transforms=False):
        self.feature_extractor.eval()
        cnn_features_list = []
        images_for_stats = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting CNN features"):
                if len(batch) == 2:
                    X, y = batch
                    labels_list.append(y.numpy())
                else:
                    X = batch

                X_device = X.to(self.device)
                # Extract features from the feature extractor
                features = self.feature_extractor(X_device)
                cnn_features_list.append(features.cpu().numpy())

                # Store denormalized images for statistical features
                images_for_stats.append(X.numpy().squeeze(1))

        cnn_features = np.concatenate(cnn_features_list, axis=0)
        images = np.concatenate(images_for_stats, axis=0)

        print("Computing statistical features...")
        stat_features = compute_statistical_features(images)

        if fit_transforms:
            stat_features = self.stat_scaler.fit_transform(stat_features)
        else:
            stat_features = self.stat_scaler.transform(stat_features)

        combined_features = np.concatenate([cnn_features, stat_features], axis=1)

        if self.use_pca:
            if fit_transforms:
                self.pca = PCA(n_components=min(self.pca_components, combined_features.shape[1]))
                combined_features = self.pca.fit_transform(combined_features)
                print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                combined_features = self.pca.transform(combined_features)

        if labels_list:
            labels = np.concatenate(labels_list, axis=0)
            return combined_features, labels
        return combined_features