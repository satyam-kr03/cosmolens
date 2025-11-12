import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..features.extractor import HybridFeatureExtractor

import xgboost as xgb


class StackedEnsemble:
    def __init__(self, cnn_model, device, xgb_params=None, pca_components=128):
        self.cnn_model = cnn_model
        self.device = device
        self.feature_extractor = HybridFeatureExtractor(cnn_model.get_feature_extractor(), device, use_pca=True,
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

        for epoch in range(1000):
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
        features = self.feature_extractor.extract_features(test_loader, fit_transforms=False)
        if isinstance(features, tuple):
            X_test = features[0]
        else:
            X_test = features

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