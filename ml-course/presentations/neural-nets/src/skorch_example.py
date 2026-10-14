# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "torch",
#   "skorch",
# ]
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cpu" }
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""
Robust Non-Linear Regression using Skorch.

Key Features:
- PEP 723 Metadata for direct execution with `uv run`.
- Skorch (PyTorch + Sklearn) for deep learning pipelines.
- Diagnostics: Log-loss history, Regression Fit, and Residuals vs Predicted.
- Functional Design: Pure functions for data and model construction.
"""

from typing import Final, Type

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

# --- Configuration & Constants ---
# System
RANDOM_SEED: Final[int] = 42

# Data Generation
N_SAMPLES: Final[int] = 1000
NOISE_LEVEL: Final[float] = 0.2
X_START: Final[float] = 0.0
X_END: Final[float] = 10.0

# Training
TEST_SIZE: Final[float] = 0.2
VALIDATION_SPLIT: Final[float] = 0.2
HIDDEN_UNITS: Final[int] = 64
MAX_EPOCHS: Final[int] = 500
LEARNING_RATE: Final[float] = 0.05
PATIENCE: Final[int] = 20


# --- 1. Domain Logic: Data Generation (Pure) ---
def generate_synthetic_data(
    n_samples: int,
    x_start: float,
    x_end: float,
    noise_level: float,
    random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic non-linear data: y = sin(x) + 0.5x + noise.

    Returns float32 arrays strictly for PyTorch compatibility.
    """
    rng = np.random.default_rng(random_seed)

    # Generate X and sort for easier visualization later
    X = np.sort(rng.uniform(x_start, x_end, n_samples)).reshape(-1, 1)

    # Ground truth function
    y_true = np.sin(X).ravel() + 0.5 * X.ravel()

    # Add Gaussian noise
    noise = rng.normal(0, noise_level, n_samples)
    y = y_true + noise

    # Cast to float32 explicitly (PyTorch standard)
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


# --- 2. Model Definition (PyTorch Module) ---
class FFNN(nn.Module):
    """
    Feed-Forward Neural Network.
    Architecture: Input -> Linear -> ReLU -> Linear -> ReLU -> Output
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- 3. Pipeline Construction ---
def build_skorch_pipeline(
    module_class: Type[nn.Module],
    max_epochs: int,
    lr: float,
    random_seed: int,
    patience: int,
    n_hidden: int
) -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline containing:
    1. StandardScaler (Normalization)
    2. NeuralNetRegressor (Skorch wrapper around PyTorch)
    """
    # Set PyTorch seed for reproducibility within the network initialization
    torch.manual_seed(random_seed)

    net = NeuralNetRegressor(
        module=module_class,
        module__n_hidden=n_hidden,
        max_epochs=max_epochs,
        lr=lr,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        # Early Stopping to prevent overfitting and populate validation history
        callbacks=[EarlyStopping(patience=patience)],
        iterator_train__shuffle=True,
    )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('net', net)
    ])


# --- 4. Visualization & Diagnostics (Side Effects) ---
def plot_diagnostics(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray
) -> None:
    """
    Generates three diagnostic plots:
    1. Training History (Log Scale)
    2. Regression Fit (Model vs Data)
    3. Residuals vs Predicted
    """
    # Extract the Skorch estimator to access history
    net = model.named_steps['net']
    history = net.history
    train_loss = history[:, 'train_loss']
    val_loss = history[:, 'valid_loss']

    # Generate predictions for the smooth fit line
    X_range = np.linspace(X_full.min(), X_full.max(), 500).reshape(-1, 1).astype(np.float32)
    y_range_pred = model.predict(X_range)

    # Calculate predictions and residuals for the Test set
    y_test_pred = model.predict(X_test)
    residuals = y_test - y_test_pred

    # Plot Setup
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Learning Curves (Log Scale)
    axes[0].plot(train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_yscale('log')  # Logarithmic Scale as requested
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MSE Loss (Log Scale)')
    axes[0].set_title('Training Convergence', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Regression Fit
    axes[1].scatter(X_full, y_full, color='black', s=5, alpha=0.1, label='Noisy Data')
    axes[1].plot(X_range, y_range_pred, color='#d62728', linewidth=3, label='Model Prediction')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('Regression Fit', fontsize=14)
    axes[1].legend()

    # Plot 3: Residuals vs Predicted
    # Diagnostic: Should look like random noise centered at 0.
    axes[2].scatter(y_test_pred, residuals, color='#9467bd', s=20, alpha=0.6)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[2].set_xlabel('Predicted Values')
    axes[2].set_ylabel('Residuals (Actual - Predicted)')
    axes[2].set_title('Residuals vs Predicted', fontsize=14)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main orchestration function."""

    # 1. Data Generation
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(N_SAMPLES, X_START, X_END, NOISE_LEVEL, RANDOM_SEED)

    # 2. Split Data
    # Note: Skorch handles the Validation split internally (default 20%) during fit.
    # We only need to split out a hold-out Test set here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # 3. Build Pipeline
    print("Building Skorch/PyTorch pipeline...")
    pipeline = build_skorch_pipeline(
        module_class=FFNN,
        max_epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        random_seed=RANDOM_SEED,
        n_hidden=HIDDEN_UNITS
    )

    # 4. Train
    print("Training model (this may take a moment)...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print(f"Final Test Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2:  {r2:.4f}")
    print("-" * 30)

    # 6. Visualize
    print("Generating plots...")
    plot_diagnostics(pipeline, X_test, y_test, X, y)


if __name__ == "__main__":
    main()
