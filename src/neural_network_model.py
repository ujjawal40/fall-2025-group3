import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from typing import Tuple, Dict, Any, Optional

from data_preprocessor import DataPreprocessor


class WideMLP(nn.Module):
    """Wide Multi-Layer Perceptron with Batch Normalization and dropout."""

    def __init__(self, n_in: int, layers: tuple, dropout_prob: float = 0.2):
        super().__init__()
        mods = []
        in_f = n_in

        for h in layers:
            mods.extend([nn.Linear(in_f, h), nn.BatchNorm1d(h), nn.ReLU()])
            if dropout_prob > 0:
                mods.append(nn.Dropout(dropout_prob))
            in_f = h

        mods.append(nn.Linear(in_f, 1))
        self.net = nn.Sequential(*mods)
    
    def forward(self, x):
        return self.net(x)


class ModelTrainer:
    """Handles model training with early stopping and learning rate scheduling."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: np.ndarray,
        test_size: float = 0.20,
        random_state: int = 42,
        n_epochs: int = 25,
        warmup_epochs: int = 3,
        batch_size: int = 512,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 5,
        hidden_layers: tuple = (128, 64, 32),
        dropout_prob: float = 0.2,
        verbose: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any], StandardScaler, np.ndarray]:
        """Train the neural network model."""

        if verbose:
            print(f"Rows for training: {X.shape[0]:,} | features: {X.shape[1]}")

        # 1. Split and scale
        price_psf = np.expm1(y)
        price_quantiles = np.quantile(price_psf, [0.5, 0.9])
        sample_weights = np.ones_like(price_psf, dtype=np.float32)
        sample_weights[price_psf >= price_quantiles[0]] = 1.2
        sample_weights[price_psf >= price_quantiles[1]] = 1.6

        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X,
            y,
            sample_weights,
            test_size=test_size,
            random_state=random_state,
        )

        # Ensure labels match the float32 tensor dtype used for training.
        X_tr_raw = X_tr.copy()
        X_val_raw = X_val.copy()

        y_tr = y_tr.astype(np.float32)
        y_val = y_val.astype(np.float32)
        w_tr = w_tr.astype(np.float32)
        w_val = w_val.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

        sqft_idx: Optional[int] = None
        sqft_val_raw: Optional[np.ndarray] = None
        if feature_names.size:
            matches = np.where(feature_names == "SQFT")[0]
            if matches.size:
                sqft_idx = int(matches[0])
                sqft_val_raw = X_val_raw[:, sqft_idx].astype(np.float32)

        # 2. DataLoaders
        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr).unsqueeze(1).float(),
                torch.from_numpy(w_tr).unsqueeze(1).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).unsqueeze(1).float(),
                torch.from_numpy(w_val).unsqueeze(1).float(),
            ),
            batch_size=batch_size,
        )

        # 3. Model
        model = WideMLP(X_tr.shape[1], hidden_layers, dropout_prob=dropout_prob).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss(reduction="none")

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs <= 0:
                warm = 1.0
            elif epoch < warmup_epochs:
                warm = (epoch + 1) / warmup_epochs
            else:
                warm = 1.0

            if epoch < warmup_epochs:
                decay = 1.0
            else:
                progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
                decay = 0.5 * (1 + math.cos(math.pi * progress))

            return warm * decay

        best_rmse, wait = math.inf, 0
        hist = {
            "rmse": [],
            "val_rmse": [],
            "val_acc_5pct": [],
            "val_acc_15pct": [],
            "epoch_sec": [],
            "lr": [],
            "val_rmse_price": [],
            "val_mae_price": [],
            "val_acc_price_5pct": [],
            "val_acc_price_15pct": [],
        }

        for epoch in range(1, n_epochs + 1):
            lr_factor = lr_lambda(epoch - 1)
            for group in opt.param_groups:
                group["lr"] = lr * lr_factor

            t0 = time.time()
            model.train()
            running = 0.0
            total_weight = 0.0

            for xb, yb, wb in tqdm(tr_loader, desc=f"Ep{epoch:02d}", leave=False):
                xb = xb.to(self.device).float()
                yb = yb.to(self.device).float()
                wb = wb.to(self.device).float()
                opt.zero_grad()
                preds = model(xb)
                per_sample = loss_fn(preds, yb) * wb
                batch_weight = wb.sum().clamp_min(1e-6)
                loss = per_sample.sum() / batch_weight
                running += per_sample.sum().item()
                total_weight += batch_weight.item()
                loss.backward()
                opt.step()

            denom = total_weight if total_weight > 0 else len(tr_loader.dataset)
            rmse_tr = math.sqrt(running / max(1e-6, denom))
            hist["rmse"].append(rmse_tr)
            hist["lr"].append(opt.param_groups[0]["lr"])

            # Validation
            model.eval()
            with torch.no_grad():
                preds = torch.cat([
                    model(xb.to(self.device))
                    for xb, _, _ in val_loader
                ]).cpu().squeeze().numpy()

            # Handle potential NaN values in predictions
            if np.isnan(preds).any():
                print(f"Warning: Found {np.isnan(preds).sum()} NaN predictions in epoch {epoch}")
                preds = np.nan_to_num(preds, nan=0.0)

            rmse_val = math.sqrt(mean_squared_error(y_val, preds))

            # Calculate percentage accuracy metrics in price-per-square-foot space
            actual_ppsqft = np.expm1(y_val)
            predicted_ppsqft = np.expm1(preds)
            pct_errors_ppsqft = (
                np.abs((actual_ppsqft - predicted_ppsqft) / actual_ppsqft) * 100
            )
            acc_5pct_ppsqft = (pct_errors_ppsqft < 5).mean() * 100
            acc_15pct_ppsqft = (pct_errors_ppsqft < 15).mean() * 100

            hist["val_rmse"].append(rmse_val)
            hist["val_acc_5pct"].append(acc_5pct_ppsqft)
            hist["val_acc_15pct"].append(acc_15pct_ppsqft)

            if sqft_val_raw is not None:
                sqft_clip = np.clip(sqft_val_raw, a_min=1.0, a_max=None)
                actual_price = actual_ppsqft * sqft_clip
                predicted_price = predicted_ppsqft * sqft_clip
                rmse_price = math.sqrt(mean_squared_error(actual_price, predicted_price))
                mae_price = mean_absolute_error(actual_price, predicted_price)
                pct_errors_price = (
                    np.abs((actual_price - predicted_price) / actual_price) * 100
                )
                acc_5pct_price = (pct_errors_price < 5).mean() * 100
                acc_15pct_price = (pct_errors_price < 15).mean() * 100
            else:
                rmse_price = float("nan")
                mae_price = float("nan")
                acc_5pct_price = float("nan")
                acc_15pct_price = float("nan")

            hist["val_rmse_price"].append(rmse_price)
            hist["val_mae_price"].append(mae_price)
            hist["val_acc_price_5pct"].append(acc_5pct_price)
            hist["val_acc_price_15pct"].append(acc_15pct_price)
            hist["epoch_sec"].append(time.time() - t0)

            if verbose:
                current_lr = opt.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:02d} | tr {rmse_tr:.4f} | val {rmse_val:.4f} | "
                    f"5%: {acc_5pct:.1f}% | 15%: {acc_15pct:.1f}% | lr {current_lr:.2e} | "
                    f"{hist['epoch_sec'][-1]:.1f}s"
                )

            if rmse_val + 1e-4 < best_rmse:
                best_rmse, wait = rmse_val, 0
                torch.save(model.state_dict(), "best_intrinsic_mlp.pt")
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print("Early stopping.")
                    break
        
        if verbose:
            print(f"Best val RMSE (log $/sqft): {best_rmse:.4f}  "
                  f"≈ $/sqft {np.expm1(best_rmse):.2f}")
            
            # Calculate final accuracy metrics
            model.load_state_dict(torch.load("best_intrinsic_mlp.pt"))
            model.eval()
            with torch.no_grad():
                final_preds = torch.cat(
                    [model(xb.to(self.device)) for xb, _, _ in val_loader]
                ).cpu().squeeze().numpy()

            actual_ppsqft = np.expm1(y_val)
            predicted_ppsqft = np.expm1(final_preds)
            pct_errors_ppsqft = (
                np.abs((actual_ppsqft - predicted_ppsqft) / actual_ppsqft) * 100
            )

            final_rmse_log = math.sqrt(mean_squared_error(y_val, final_preds))
            final_mae_log = mean_absolute_error(y_val, final_preds)

            final_rmse_ppsqft = math.sqrt(
                mean_squared_error(actual_ppsqft, predicted_ppsqft)
            )
            final_mae_ppsqft = mean_absolute_error(actual_ppsqft, predicted_ppsqft)

            final_acc_5pct_ppsqft = (pct_errors_ppsqft < 5).mean() * 100
            final_acc_15pct_ppsqft = (pct_errors_ppsqft < 15).mean() * 100

            if sqft_val_raw is not None:
                sqft_clip = np.clip(sqft_val_raw, a_min=1.0, a_max=None)
                actual_price = actual_ppsqft * sqft_clip
                predicted_price = predicted_ppsqft * sqft_clip
                pct_errors_price = (
                    np.abs((actual_price - predicted_price) / actual_price) * 100
                )
                final_rmse_price = math.sqrt(
                    mean_squared_error(actual_price, predicted_price)
                )
                final_mae_price = mean_absolute_error(actual_price, predicted_price)
                final_acc_5pct_price = (pct_errors_price < 5).mean() * 100
                final_acc_15pct_price = (pct_errors_price < 15).mean() * 100
            else:
                final_rmse_price = float("nan")
                final_mae_price = float("nan")
                final_acc_5pct_price = float("nan")
                final_acc_15pct_price = float("nan")

            print(f"Final Accuracy Metrics:")
            print(f"  < 5% error (PPSQFT): {final_acc_5pct_ppsqft:.2f}%")
            print(f"  < 15% error (PPSQFT): {final_acc_15pct_ppsqft:.2f}%")
            if not np.isnan(final_acc_5pct_price):
                print(f"  < 5% error (Price): {final_acc_5pct_price:.2f}%")
                print(f"  < 15% error (Price): {final_acc_15pct_price:.2f}%")
            print("Final Error Metrics (log space):")
            print(f"  RMSE: {final_rmse_log:.4f}")
            print(f"  MAE: {final_mae_log:.4f}")
            print("Final Error Metrics (price-per-sqft):")
            print(f"  RMSE: ${final_rmse_ppsqft:,.2f}")
            print(f"  MAE: ${final_mae_ppsqft:,.2f}")
            if not np.isnan(final_rmse_price):
                print("Final Error Metrics (total price):")
                print(f"  RMSE: ${final_rmse_price:,.2f}")
                print(f"  MAE: ${final_mae_price:,.2f}")
            print(f"Target (WITHOUT Loc): < 5% = 24.60%, < 15% = 64.54%")
            print(f"Target (WITH Loc): < 5% = 27.43%, < 15% = 70.10%")

            if final_acc_5pct_ppsqft > 24.60:
                print(f"✅ BEAT WITHOUT Loc target for < 5%!")
            else:
                print(f"❌ Did NOT beat WITHOUT Loc target for < 5%")

            if final_acc_15pct_ppsqft > 64.54:
                print(f"✅ BEAT WITHOUT Loc target for < 15%!")
            else:
                print(f"❌ Did NOT beat WITHOUT Loc target for < 15%")
        
        model.load_state_dict(torch.load("best_intrinsic_mlp.pt"))
        return model, hist, scaler, feature_names


def main():
    """Main function to execute the complete pipeline."""
    print("Starting House Price Prediction Pipeline...")

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Load and clean data
    print("\n1. Loading and preprocessing data...")
    df = preprocessor.load_data('sub_sample.csv')

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Original data shape: {df.shape}")
    print(f"Columns available: {list(df.columns)}")

    # Clean and engineer features
    print("\n2. Cleaning and engineering features...")
    clean_df = preprocessor.clean_and_engineer(
        df,
        one_hot=True,
        max_categories=50,
        min_frequency=0.01,
    )
    print(f"Cleaned data shape: {clean_df.shape}")

    # Prepare features for training
    print("\n3. Preparing features for training...")
    X, y, feature_names = preprocessor.prepare_features(clean_df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train model
    print("\n4. Training neural network model...")
    model, history, scaler, feat_names = trainer.train_model(
        X, y, feature_names,
        n_epochs=160,
        batch_size=256,
        lr=5e-4,
        weight_decay=5e-4,
        patience=20,
        warmup_epochs=8,
        hidden_layers=(256, 128, 64, 32),
        dropout_prob=0.25,
        verbose=True
    )
    
    print("\n5. Training completed!")
    print(f"Model saved as 'best_intrinsic_mlp.pt'")
    print(f"Feature scaler and names available for predictions")
    
    # Plot training history
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.plot(history["rmse"], label="Training RMSE")
    plt.plot(history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(history["val_acc_5pct"], label="< 5% Accuracy", color='green')
    plt.axhline(y=24.60, color='red', linestyle='--', label='Target (WITHOUT Loc)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("5% Error Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(history["val_acc_15pct"], label="< 15% Accuracy", color='blue')
    plt.axhline(y=64.54, color='red', linestyle='--', label='Target (WITHOUT Loc)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("15% Error Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(history["lr"], label="Learning Rate", color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history, scaler, feat_names


if __name__ == "__main__":
    model, history, scaler, feat_names = main()
