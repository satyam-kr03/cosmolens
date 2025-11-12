# FAIR Universe - [Weak Lensing ML Uncertainty Challenge](https://www.codabench.org/competitions/8934/)

This NeurIPS 2025 Machine Learning competition explores uncertainty-aware and out-of-distribution detection AI techniques for Weak Gravitational Lensing Cosmology.

## Project Structure

```
cosmolens/
├── src/
│   ├── utils/           # Utility classes and data handling
│   │   ├── utility.py   # Utility functions
│   │   ├── data.py      # Data loading and preprocessing
│   │   └── score.py     # Scoring functions
│   ├── models/          # Neural network architectures
│   │   ├── resnet.py    # Multi-scale ResNet implementation
│   │   └── ensemble.py  # XGBoost ensemble model
│   ├── features/        # Feature extraction modules
│   │   └── extractor.py # Hybrid CNN + statistical features
│   ├── training/        # Training utilities
│   │   ├── dataset.py   # PyTorch dataset classes
│   │   ├── train.py     # Training and validation functions
│   │   └── config.py    # Configuration parameters
│   └── inference/       # Inference and prediction
│       └── pipeline.py  # Improved prediction pipeline
├── models/              # Saved model checkpoints
├── notebooks/           # Experimental Jupyter notebooks
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
└── README.md           
```

### Key Modules

- **`src.utils`**: Data loading, utility functions, and scoring
- **`src.models`**: Neural network architectures (ResNet, Ensemble)
- **`src.features`**: Feature extraction combining CNN and statistical features
- **`src.training`**: Dataset classes, training loops, and configuration
- **`src.inference`**: Advanced prediction pipeline with TTA and MCMC

### Installation & Usage

Install dependencies
```bash
pip install -r requirements.txt
```

Run the main pipeline
```bash
python main.py
```

---

# Approach: A Hybrid ResNet-XGBoost Ensemble 

Here we outline a machine learning approach for estimating the cosmological parameters $\Omega_m$ (Omega matter) and $S_8$ from weak-lensing convergence maps (kappa maps). The solution employs a sophisticated stacked ensemble model that combines a Convolutional Neural Network (CNN) for feature extraction with an XGBoost model for regression, further refined by a neural network meta-learner. The final parameter uncertainties are derived using a Markov Chain Monte Carlo (MCMC) method informed by the model's predictions.

## 1. Data Preprocessing and Augmentation

1.  **Data Loading**: The training (`kappa`) and test (`kappa_test`) convergence maps are loaded as NumPy arrays. A binary mask is used to handle the specific geometry of the survey area.
2.  **Train/Validation Split**: The full training dataset, which consists of 101 cosmologies with 256 systematic variations each, is split into training and validation sets (e.g., `noisy_kappa_train`, `noisy_kappa_val`). The labels ($\Omega_m$, $S_8$) are also loaded and similarly split.
3.  **Normalization**: The labels (parameters) are standardized using `sklearn.preprocessing.StandardScaler` for training, and the image data is normalized (mean/std) via `torchvision.transforms` before being fed into the neural network.
4.  **Data Augmentation**: To improve model robustness and prevent overfitting, the training dataset is augmented in real-time. The `AugmentedCosmologyDataset` class applies random augmentations during training, including:
    * Horizontal flips (`np.fliplr`)
    * Vertical flips (`np.flipud`)
    * 180-degree rotations (`np.rot90`)
    * Addition of small Gaussian noise.

## 2. Model Architecture: A Stacked Ensemble

The core of this approach is a `StackedEnsemble` model that combines the strengths of deep learning and gradient boosting in a two-level architecture.

### Level 0: Feature Extractors and Base Learners

1.  **CNN Feature Extractor (`MultiScaleResNet`)**:
    * A custom ResNet-based architecture is used as the primary feature extractor.
    * It begins with an initial convolutional layer and max pooling to reduce spatial dimensions.
    * It then passes the data through four sequential `ResidualBlock` layers, progressively increasing the feature depth (64 -> 128 -> 256 -> 512) and downsampling the map.
    * To capture information at multiple scales, the output of *each* of the four residual layers is passed through an `AdaptiveAvgPool2d` layer, flattened, and concatenated.
    * This multi-scale feature vector is fed into a final fully connected (FC) block with Dropout and BatchNorm to produce a 256-dimensional feature vector.

2.  **Statistical Feature Extractor**:
    * A separate function, `compute_statistical_features`, calculates a set of 8 classical statistics for each map (mean, std, skew, kurtosis, 25th/75th percentiles, min, and max). These statistical features are standardized using `StandardScaler`.

3.  **Hybrid Feature Extractor**:
    * The 256 features from the `MultiScaleResNet` are combined with the 8 statistical features.
    * Principal Component Analysis (`PCA`) is then applied to this combined feature vector for dimensionality reduction (e.g., to 128 components). This resulting vector serves as the "hybrid feature" input for the XGBoost models.

4.  **Base Learners**:
    * **XGBoost Models**: Two separate `xgboost.XGBRegressor` models are trained—one for $\Omega_m$ and one for $S_8$. Both are trained on the same *hybrid feature* vector.
    * **CNN Prediction Head**: The original CNN (`MultiScaleResNetWithHead`) also has its own linear prediction head that produces direct estimates for $\Omega_m$ and $S_8$ from the 256-dimensional CNN feature vector.

### Level 1: Meta-Learner (Stacking)

The predictions from the base learners are used as features to train a final meta-learner:

1.  **Feature Creation**: For the training set, predictions are gathered from:
    * The CNN's prediction head (2 predictions: $\Omega_m, S_8$).
    * The trained $\Omega_m$-XGBoost model (1 prediction).
    * The trained $S_8$-XGBoost model (1 prediction).
2.  **Meta-Model**: These 4 predictions are concatenated into a new feature vector. This vector is used to train a simple feed-forward neural network (Linear -> ReLU -> Dropout -> Linear) that outputs the final two parameter estimates.

## 3. Training and Hyperparameter Optimization

1.  **CNN Feature Extractor Training**: The `MultiScaleResNetWithHead` is trained first as a standalone model. It is trained for 15 epochs using `MSELoss`, an `AdamW` optimizer, and a `CosineAnnealingWarmRestarts` learning rate scheduler. Early stopping with a patience of 7 epochs is used to save the best model based on validation loss.
2.  **Ensemble Training**: After the CNN is trained and its weights are frozen, the `StackedEnsemble` is fit. This involves (a) extracting all hybrid features for the training data, (b) training the two XGBoost models, and (c) training the meta-learner for 1000 epochs.
3.  **Hyperparameter Optimization (Optuna)**: The `Optuna` library is used to perform a 5-trial optimization search for the best hyperparameters. This search tunes a wide array of parameters simultaneously, including the CNN's feature dimension and dropout, the optimizer's learning rate and weight decay, batch size, PCA components, and all key XGBoost parameters (e.g., `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.). The objective is to minimize the final MSE on the validation set.

## 4. Prediction and Uncertainty Estimation

The final predictions for the competition require both a point estimate and an error bar.

1.  **Test-Time Augmentation (TTA)**: For generating the final predictions, the `ImprovedPredictionPipeline` uses TTA. It generates 8 augmented versions (flips, rotations, etc.) of each test map, runs each through the full `StackedEnsemble`, and averages the 8 resulting predictions. This significantly improves the robustness and accuracy of the final estimate.
2.  **MCMC for Uncertainty**: The model's point estimates are used to build a likelihood model for a Markov Chain Monte Carlo (MCMC) sampler.
    * First, the mean prediction (`mean_d_vector`) and covariance matrix (`cov_d_vector`) of the ensemble's predictions are calculated for each of the 101 known cosmologies in the validation set.
    * `LinearNDInterpolator` (and later, `RBFInterpolator` for improved smoothness) is used to create functions that can interpolate the mean and covariance for *any* given ($\Omega_m, S_8$) pair.
    * These interpolators are used to define a log-likelihood function (`loglike`) that calculates the probability of the model's TTA prediction *given* a true set of parameters.
    * For each of the 4000 test maps, 4 independent MCMC chains are run for 12,000 steps each to sample the posterior distribution, with an adaptive step size to optimize acceptance rates.
    * The first 25% of samples from each chain are discarded as burn-in. The remaining samples from all chains are combined.
    * The **mean** of this combined posterior distribution is used as the final point estimate ($\Omega_m$, $S_8$), and the **standard deviation** is used as the uncertainty (error bar).
3.  **Error Calibration**: Finally, the error bars from the validation set MCMC are compared to the actual errors (predicted vs. true values) to compute a "calibration factor". This factor is used to scale the final test set error bars to ensure they are well-calibrated for the competition's scoring function.