# %%
import os
import json
import time
import zipfile
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import kurtosis, skew

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import optuna

# %matplotlib inline

# %%
class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        return data + np.random.randn(*data.shape) * 0.4 / (2*ng*pixel_size**2)**0.5 * mask
    
    @staticmethod
    def load_np(data_dir, file_name):
        return np.load(os.path.join(data_dir, file_name))

    @staticmethod
    def save_np(data_dir, file_name, data):
        np.save(os.path.join(data_dir, file_name), data)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        os.makedirs(submission_dir, exist_ok=True)
        json_path = os.path.join(submission_dir, json_file_name)
        with open(json_path, "w") as f:
            json.dump(data, f)
        zip_path = os.path.join(submission_dir, zip_file_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)
        os.remove(json_path)
        return zip_path


class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        self.viz_label_file = 'label.npy'
        
        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = self.viz_label_file
            self.Ncosmo = 101
            self.Nsys = 256
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3
            self.Nsys = 30
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3
        
        self.shape = [1424, 176]
        self.pixelsize_arcmin = 2
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi
        self.ng = 30

    def load_train_data(self):
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file)
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:,:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.kappa_file)
        self.label = Utility.load_np(data_dir=self.data_dir, file_name=self.label_file)
        self.viz_label = Utility.load_np(data_dir=self.data_dir, file_name=self.viz_label_file)

    def load_test_data(self):
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.test_kappa_file)


class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        return score if score >= -10**6 else -10**6

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MultiScaleResNet(nn.Module):
    def __init__(self, height, width, feature_dim=256, dropout=0.3):
        super(MultiScaleResNet, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(64 + 128 + 256 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        f1 = self.avgpool1(x1).view(x1.size(0), -1)
        f2 = self.avgpool2(x2).view(x2.size(0), -1)
        f3 = self.avgpool3(x3).view(x3.size(0), -1)
        f4 = self.avgpool4(x4).view(x4.size(0), -1)
        
        multi_scale = torch.cat([f1, f2, f3, f4], dim=1)
        features = self.fc(multi_scale)
        
        return features


class MultiScaleResNetWithHead(nn.Module):
    def __init__(self, height, width, num_targets, feature_dim=256, dropout=0.3):
        super(MultiScaleResNetWithHead, self).__init__()
        self.feature_extractor = MultiScaleResNet(height, width, feature_dim, dropout)
        self.prediction_head = nn.Linear(feature_dim, num_targets)
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        if return_features:
            return features
        predictions = self.prediction_head(features)
        return predictions
    
    def get_feature_extractor(self):
        return self.feature_extractor

# %%
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
    def __init__(self, cnn_model, device, use_pca=True, pca_components=128):
        self.cnn_model = cnn_model
        self.device = device
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self.stat_scaler = StandardScaler()
        
    def extract_features(self, dataloader, fit_transforms=False):
        self.cnn_model.eval()
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
                features = self.cnn_model(X_device)
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
                self.pca = PCA(n_components=self.pca_components)
                combined_features = self.pca.fit_transform(combined_features)
                print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                combined_features = self.pca.transform(combined_features)
        
        if labels_list:
            labels = np.concatenate(labels_list, axis=0)
            return combined_features, labels
        return combined_features

# %%
class StackedEnsemble:
    def __init__(self, cnn_model, device, xgb_params=None, pca_components=128):
        self.cnn_model = cnn_model
        self.device = device
        self.feature_extractor = HybridFeatureExtractor(cnn_model, device, use_pca=True, 
                                                        pca_components=pca_components)
        
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        self.xgb_params = xgb_params
        self.xgb_models = []
        self.meta_learner = None
        
    def fit(self, train_loader, num_targets=2):
        print("Extracting training features...")
        X_train, y_train = self.feature_extractor.extract_features(train_loader, fit_transforms=True)
        
        print(f"Training XGBoost models on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.xgb_models = []
        for i in range(num_targets):
            print(f"Training XGBoost for target {i+1}/{num_targets}...")
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train[:, i], verbose=False)
            self.xgb_models.append(model)
        
        print("Training meta-learner (stacking)...")
        self.cnn_model.eval()
        cnn_preds_list = []
        with torch.no_grad():
            for batch in train_loader:
                X, _ = batch
                X = X.to(self.device)
                preds = self.cnn_model.prediction_head(self.cnn_model.feature_extractor(X))
                cnn_preds_list.append(preds.cpu().numpy())
        cnn_preds = np.concatenate(cnn_preds_list, axis=0)
        
        xgb_preds = np.column_stack([model.predict(X_train) for model in self.xgb_models])
        
        stacked_features = np.concatenate([cnn_preds, xgb_preds], axis=1)
        
        self.meta_learner = nn.Sequential(
            nn.Linear(num_targets * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_targets)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        X_meta = torch.FloatTensor(stacked_features).to(self.device)
        y_meta = torch.FloatTensor(y_train).to(self.device)
        
        for epoch in range(100):
            self.meta_learner.train()
            optimizer.zero_grad()
            preds = self.meta_learner(X_meta)
            loss = loss_fn(preds, y_meta)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f"Meta-learner epoch {epoch+1}/100, Loss: {loss.item():.6f}")
        
        print("Ensemble training complete!")
        
    def predict(self, test_loader):
        print("Extracting test features...")
        X_test = self.feature_extractor.extract_features(test_loader, fit_transforms=False)
        
        print("Making XGBoost predictions...")
        xgb_preds = np.column_stack([model.predict(X_test) for model in self.xgb_models])
        
        print("Making CNN predictions...")
        self.cnn_model.eval()
        cnn_preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, list):
                    X = batch[0]
                else:
                    X = batch
                X = X.to(self.device)
                preds = self.cnn_model.prediction_head(self.cnn_model.feature_extractor(X))
                cnn_preds_list.append(preds.cpu().numpy())
        cnn_preds = np.concatenate(cnn_preds_list, axis=0)
        
        print("Combining predictions with meta-learner...")
        stacked_features = np.concatenate([cnn_preds, xgb_preds], axis=1)
        X_meta = torch.FloatTensor(stacked_features).to(self.device)
        
        self.meta_learner.eval()
        with torch.no_grad():
            final_preds = self.meta_learner(X_meta).cpu().numpy()
        
        return final_preds

# %%
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        pred = model(X, return_features=False)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X, return_features=False)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    return total_loss / len(dataloader)


class AugmentedCosmologyDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, augment=False):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)
        
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
            k = np.random.randint(0, 2)*2 # only 0 or 180 degrees to preserve shape
            if k > 0:
                image = np.rot90(image, k).copy()
            if np.random.rand() > 0.7:
                noise = np.random.randn(*image.shape) * 0.01
                image = image + noise
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            label = torch.from_numpy(label)
            return image, label
        else:
            return image

# %%
root_dir = os.getcwd()
USE_PUBLIC_DATASET = True
PUBLIC_DATA_DIR = '../data'
DATA_DIR = PUBLIC_DATA_DIR if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')

class Config:
    NUM_TARGETS = 2
    FEATURE_DIM = 256
    DROPOUT = 0.35
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    PCA_COMPONENTS = 128
    
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CNN_MODEL_PATH = os.path.join(root_dir, "improved_cnn_feature_extractor.pth")
    ENSEMBLE_PATH = os.path.join(root_dir, "improved_ensemble.pkl")

config = Config()
print(f"Device: {config.DEVICE}")

# %%
data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
data_obj.load_train_data()
data_obj.load_test_data()

print(f"Train shape: {data_obj.kappa.shape}, Test shape: {data_obj.kappa_test.shape}")

# %%
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

# %%
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

# %%
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

# %%
USE_OPTUNA = False
N_TRIALS = 30

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

# %%
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
            print(f"  ✓ Best model saved!")
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

# %%
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

# %%
print("Predicting on validation set...")
y_pred_val = ensemble.predict(val_loader)
y_pred_val = label_scaler.inverse_transform(y_pred_val)

print(f"Predictions: {y_pred_val.shape}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_val[:,0], y_pred_val[:,0], alpha=0.5, s=15)
axes[0].plot(sorted(y_val[:,0]), sorted(y_val[:,0]), 'k--', linewidth=2)
axes[0].set_xlabel('Ground Truth')
axes[0].set_ylabel('Prediction')
axes[0].set_title(r'$\Omega_m$')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_val[:,1], y_pred_val[:,1], alpha=0.5, s=15)
axes[1].plot(sorted(y_val[:,1]), sorted(y_val[:,1]), 'k--', linewidth=2)
axes[1].set_xlabel('Ground Truth')
axes[1].set_ylabel('Prediction')
axes[1].set_title(r'$S_8$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

mse_om = mean_squared_error(y_val[:,0], y_pred_val[:,0])
mse_s8 = mean_squared_error(y_val[:,1], y_pred_val[:,1])
r2_om = r2_score(y_val[:,0], y_pred_val[:,0])
r2_s8 = r2_score(y_val[:,1], y_pred_val[:,1])

print(f"Ωₘ - MSE: {mse_om:.6f}, R²: {r2_om:.4f}")
print(f"S₈ - MSE: {mse_s8:.6f}, R²: {r2_s8:.4f}")

# %%
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

# %%
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

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].errorbar(y_val[:,0], mean_val[:,0], yerr=errorbar_val[:,0],
                 fmt='o', capsize=2, alpha=0.5, markersize=3)
axes[0].plot(sorted(y_val[:,0]), sorted(y_val[:,0]), 'k--', linewidth=2)
axes[0].set_xlabel('Ground Truth')
axes[0].set_ylabel('Prediction')
axes[0].set_title(r'$\Omega_m$ with MCMC')
axes[0].grid(True, alpha=0.3)

axes[1].errorbar(y_val[:,1], mean_val[:,1], yerr=errorbar_val[:,1],
                 fmt='o', capsize=2, alpha=0.5, markersize=3)
axes[1].plot(sorted(y_val[:,1]), sorted(y_val[:,1]), 'k--', linewidth=2)
axes[1].set_xlabel('Ground Truth')
axes[1].set_ylabel('Prediction')
axes[1].set_title(r'$S_8$ with MCMC')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

validation_score = Score._score_phase1(y_val, mean_val, errorbar_val)
print(f"\nValidation Score: {validation_score:.2f}")
print(f"Error bar (Ωₘ): {np.mean(errorbar_val[:, 0]):.6f}")
print(f"Error bar (S₈): {np.mean(errorbar_val[:, 1]):.6f}")

# %%
print("Predicting on test set...")
y_pred_test = ensemble.predict(test_loader)
y_pred_test = label_scaler.inverse_transform(y_pred_test)

print(f"Test predictions: {y_pred_test.shape}")

# %%
Nstep_test = 10000
sigma_test = 0.06

current_test = cosmology[np.random.choice(Ncosmo, size=data_obj.Ntest)]
curr_logprob_test = logp_posterior(current_test, y_pred_test)

states_test = []
total_acc_test = np.zeros(len(current_test))

print("Running MCMC on test set...")

for i in tqdm(range(Nstep_test), desc="MCMC Test"):
    proposal = current_test + np.random.randn(*current_test.shape) * sigma_test
    proposal_logprob = logp_posterior(proposal, y_pred_test)
    
    acc_logprob = proposal_logprob - curr_logprob_test
    acc_logprob[acc_logprob > 0] = 0
    acc_prob = np.exp(acc_logprob)
    acc = np.random.uniform(size=len(current_test)) < acc_prob
    
    total_acc_test += acc_prob
    current_test[acc] = proposal[acc]
    curr_logprob_test[acc] = proposal_logprob[acc]
    states_test.append(np.copy(current_test)[None])

states_test = np.concatenate(states_test[int(0.2*Nstep_test):], 0)
mean_test = np.mean(states_test, 0)
errorbar_test = np.std(states_test, 0)

print(f"MCMC complete! Acceptance rate: {np.mean(total_acc_test/Nstep_test):.3f}")
print(f"Mean error bars: {np.mean(errorbar_test, 0)}")

# %%
data_submission = {
    "means": mean_test.tolist(),
    "errorbars": errorbar_test.tolist()
}

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
zip_file_name = f'Submission_Improved_{timestamp}.zip'

zip_file = Utility.save_json_zip(
    submission_dir="submissions",
    json_file_name="result.json",
    zip_file_name=zip_file_name,
    data=data_submission
)

print("=" * 70)
print("SUBMISSION CREATED")
print("=" * 70)
print(f"File: {zip_file}")
print(f"Test samples: {len(mean_test)}")
print("=" * 70)


