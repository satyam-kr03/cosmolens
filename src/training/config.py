import os
import torch


class Config:
    NUM_TARGETS = 2
    FEATURE_DIM = 256
    DROPOUT = 0.35
    BATCH_SIZE = 32
    EPOCHS = 15
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

    # Model paths
    CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/improved_cnn_feature_extractor.pth")
    ENSEMBLE_PATH = os.path.join(os.path.dirname(__file__), "../../models/improved_ensemble.pkl")