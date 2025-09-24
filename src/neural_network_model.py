import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
import time
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from typing import Tuple, Dict, Any, Optional


class DataPreprocessor:
    """Handles data loading, cleaning, and feature engineering for house price prediction."""
    
    def __init__(self):
        # Define column groups for feature engineering
        self.COLS_1A = [
            "SQFT", "BEDROOMS", "BATHROOMS", "STORIES", "LEVELS", "LOT",
            "PARKING", "POOLFEATURES", "BASEMENT", "STRUCTURETYPE",
            "HOMETYPE", "PROPERTYCONDITION", "COOLINGFEATURES",
            "HEATINGFEATURES", "SENIORLIVING", "NEWCONSTRUCTIONFLAG",
        ]
        
        self.COLS_1B = ["YEARBUILT", "CREATEDAT_YEAR", "CREATEDAT_MONTH"]
        
        self.COLS_1C = [
            "ELEMNTARYSCHOOLRATING", "MIDDLESCHOOLRATING", "HIGHSCHOOLRATING",
            "MONTHLY_UNEMPLOYMENT_RATE", "MONTHLY_AVG_MORTGAGE_RATE",
            "HOTNESS_SCORE", "SUPPLY_SCORE", "DEMAND_SCORE",
            "MEDIAN_DAYS_ON_MARKET",
        ]
        
        self.COLS_1D = ["STATE_FIPS", "COUNTY_FIPS"]
        
        self.KEEP_COLS = ["ZPID"] + self.COLS_1A + self.COLS_1B + self.COLS_1C + self.COLS_1D + ["PRICE"]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded data shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return pd.DataFrame()
    
    def num_from_text(self, series: pd.Series, allow_comma: bool = True) -> pd.Series:
        """Extract the first number from strings like '1,393 sqft' → 1393.0"""
        pattern = r"([-+]?\d[\d,]*\.?\d*)"
        cleaned = series.astype(str).str.extract(pattern, expand=False)
        
        if allow_comma:
            cleaned = cleaned.str.replace(",", "", regex=False)
        
        return pd.to_numeric(cleaned, errors="coerce")
    
    def clean_and_engineer(self, df: pd.DataFrame, one_hot: bool = True) -> pd.DataFrame:
        """Clean and engineer features from raw data."""
        df = df.copy()
        
        # 1. Text-to-number for obvious numeric fields
        numeric_text_cols = [
            "BEDROOMS", "BATHROOMS", "STORIES", "LEVELS",
            "LOT", "PARKING", "PARKINGTOTALSPACES", "YEARBUILT"
        ]
        
        for col in numeric_text_cols:
            if col in df.columns:
                df[col] = self.num_from_text(df[col])
        
        # LOT: convert small (<10) acres → sqft
        if "LOT" in df.columns:
            mask = df["LOT"] < 10
            df.loc[mask, "LOT"] = df.loc[mask, "LOT"] * 43_560
        
        # Unify parking column
        raw_park = "PARKING" if "PARKING" in df.columns else "PARKINGTOTALSPACES"
        if raw_park in df.columns:
            df["GARAGE_SPACES"] = df[raw_park]
            df.drop(columns=[raw_park], inplace=True)
        
        # Force numeric dtypes
        for col in ["BEDROOMS", "BATHROOMS", "STORIES", "LEVELS",
                    "GARAGE_SPACES", "YEARBUILT"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 2. Binary Yes/No flags
        yn = {"Yes": 1, "No": 0}
        if "NEWCONSTRUCTIONFLAG" in df.columns:
            df["NEWCONSTRUCTIONFLAG"] = df["NEWCONSTRUCTIONFLAG"].map(yn)
        if "SENIORLIVING" in df.columns:
            df["SENIORLIVING"] = df["SENIORLIVING"].map(yn)
        
        # 3. Derived numerics
        if "CREATEDAT_YEAR" in df.columns and "YEARBUILT" in df.columns:
            df["PROPERTY_AGE"] = df["CREATEDAT_YEAR"] - df["YEARBUILT"]
            df.loc[df["PROPERTY_AGE"] < 0, "PROPERTY_AGE"] = np.nan
        
        # Handle zero/negative values
        for col in ["SQFT", "LOT", "MEDIAN_DAYS_ON_MARKET", "PRICE"]:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan
        
        # Log transformations
        if "SQFT" in df.columns:
            df["LOG_SQFT"] = np.log1p(df["SQFT"])
        if "LOT" in df.columns:
            df["LOG_LOT"] = np.log1p(df["LOT"])
        if "MEDIAN_DAYS_ON_MARKET" in df.columns:
            df["LOG_DOM"] = np.log1p(df["MEDIAN_DAYS_ON_MARKET"])
        
        # Cyclical encoding for months
        if "CREATEDAT_MONTH" in df.columns:
            two_pi = 2 * np.pi
            df["MONTH_SIN"] = np.sin(two_pi * df["CREATEDAT_MONTH"] / 12)
            df["MONTH_COS"] = np.cos(two_pi * df["CREATEDAT_MONTH"] / 12)
        
        # 4. Imputations + missing flags
        base_imp = ["SQFT", "LOT", "GARAGE_SPACES", "PROPERTY_AGE"]
        for col in base_imp:
            if col in df.columns:
                df[f"MISS_{col}"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())
        
        # Neighborhood features imputation
        neigh_cols = [
            "HOTNESS_SCORE", "SUPPLY_SCORE", "DEMAND_SCORE",
            "MONTHLY_UNEMPLOYMENT_RATE", "MONTHLY_AVG_MORTGAGE_RATE",
            "MEDIAN_DAYS_ON_MARKET"
        ]
        for col in neigh_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # 5. Handle remaining object columns
        remain_obj = df.select_dtypes("object").columns
        truly_cat = []
        
        for col in remain_obj:
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().mean() >= 0.80:  # mostly numeric
                df[col] = conv
            else:
                truly_cat.append(col)
        
        if one_hot and truly_cat:
            df = pd.get_dummies(df, columns=truly_cat, dummy_na=True, prefix_sep="==")
        else:
            df.drop(columns=truly_cat, inplace=True)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and target for training."""
        # Filter valid data
        df = df[(df["PRICE"] > 0) & (df["SQFT"] > 0)].copy()
        
        # Create price per square foot target
        df["PPSQFT"] = df["PRICE"] / df["SQFT"].clip(lower=1)
        df["LOG_PPSQFT"] = np.log1p(df["PPSQFT"])
        
        # Remove rows with invalid target values
        df = df[np.isfinite(df["LOG_PPSQFT"])].copy()
        
        # Prepare features
        id_cols = ["ZPID"]
        drop_cols = ["PRICE", "PPSQFT", "LOG_PPSQFT"] + [c for c in id_cols if c in df.columns]
        
        X_df = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors="ignore")
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # More robust median calculation
        for col in X_df.columns:
            if X_df[col].isna().all():
                X_df[col] = 0.0
            else:
                X_df[col] = X_df[col].fillna(X_df[col].median())
        
        X_df = X_df.astype(np.float32)
        
        # Final check for any remaining NaN/Inf values
        X_df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)
        
        X = X_df.values
        y = df["LOG_PPSQFT"].values.astype(np.float32)
        feature_names = X_df.columns.values
        
        # Validate final arrays
        assert not np.isnan(X).any(), "Features contain NaN values"
        assert not np.isnan(y).any(), "Target contains NaN values"
        assert np.isfinite(X).all(), "Features contain infinite values"
        assert np.isfinite(y).all(), "Target contains infinite values"
        
        return X, y, feature_names


class WideMLP(nn.Module):
    """Wide Multi-Layer Perceptron with Batch Normalization."""
    
    def __init__(self, n_in: int, layers: tuple):
        super().__init__()
        mods = []
        in_f = n_in
        
        for h in layers:
            mods += [nn.Linear(in_f, h), nn.BatchNorm1d(h), nn.ReLU()]
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
        verbose: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any], StandardScaler, np.ndarray]:
        """Train the neural network model."""
        
        if verbose:
            print(f"Rows for training: {X.shape[0]:,} | features: {X.shape[1]}")
        
        # 1. Split and scale
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        
        # 2. DataLoaders
        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)),
            batch_size=batch_size
        )
        
        # 3. Model
        model = WideMLP(X_tr.shape[1], hidden_layers).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        
        def lr_now(ep):  # linear warm-up then flat
            return lr * (ep + 1) / warmup_epochs if ep < warmup_epochs else lr
        
        best_rmse, wait = math.inf, 0
        hist = {"rmse": [], "val_rmse": [], "val_acc_5pct": [], "val_acc_15pct": [], "epoch_sec": []}
        
        for epoch in range(1, n_epochs + 1):
            for g in opt.param_groups:
                g["lr"] = lr_now(epoch - 1)
            
            t0 = time.time()
            model.train()
            running = 0.0
            
            for xb, yb in tqdm(tr_loader, desc=f"Ep{epoch:02d}", leave=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                running += loss.mul(len(xb)).item()
                loss.backward()
                opt.step()
            
            rmse_tr = math.sqrt(running / len(tr_loader.dataset))
            hist["rmse"].append(rmse_tr)
            
            # Validation
            model.eval()
            with torch.no_grad():
                preds = torch.cat([model(xb.to(self.device)) for xb, _ in val_loader]).cpu().squeeze().numpy()
            
            # Handle potential NaN values in predictions
            if np.isnan(preds).any():
                print(f"Warning: Found {np.isnan(preds).sum()} NaN predictions in epoch {epoch}")
                preds = np.nan_to_num(preds, nan=0.0)
            
            rmse_val = math.sqrt(mean_squared_error(y_val, preds))
            
            # Calculate percentage accuracy metrics
            actual_prices = np.expm1(y_val)  # Convert back from log space
            predicted_prices = np.expm1(preds)
            
            # Calculate percentage errors
            pct_errors = np.abs((actual_prices - predicted_prices) / actual_prices) * 100
            
            # Calculate accuracy within thresholds
            acc_5pct = (pct_errors < 5).mean() * 100
            acc_15pct = (pct_errors < 15).mean() * 100
            
            hist["rmse"].append(rmse_tr)
            hist["val_rmse"].append(rmse_val)
            hist["val_acc_5pct"].append(acc_5pct)
            hist["val_acc_15pct"].append(acc_15pct)
            hist["epoch_sec"].append(time.time() - t0)
            
            if verbose:
                print(f"Epoch {epoch:02d} | tr {rmse_tr:.4f} | val {rmse_val:.4f} | "
                      f"5%: {acc_5pct:.1f}% | 15%: {acc_15pct:.1f}% | {hist['epoch_sec'][-1]:.1f}s")
            
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
                final_preds = torch.cat([model(xb.to(self.device)) for xb, _ in val_loader]).cpu().squeeze().numpy()
            
            actual_prices = np.expm1(y_val)
            predicted_prices = np.expm1(final_preds)
            pct_errors = np.abs((actual_prices - predicted_prices) / actual_prices) * 100
            
            final_acc_5pct = (pct_errors < 5).mean() * 100
            final_acc_15pct = (pct_errors < 15).mean() * 100
            
            print(f"Final Accuracy Metrics:")
            print(f"  < 5% error: {final_acc_5pct:.2f}%")
            print(f"  < 15% error: {final_acc_15pct:.2f}%")
            print(f"Target (WITHOUT Loc): < 5% = 24.60%, < 15% = 64.54%")
            print(f"Target (WITH Loc): < 5% = 27.43%, < 15% = 70.10%")
            
            if final_acc_5pct > 24.60:
                print(f"✅ BEAT WITHOUT Loc target for < 5%!")
            else:
                print(f"❌ Did NOT beat WITHOUT Loc target for < 5%")
                
            if final_acc_15pct > 64.54:
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
    clean_df = preprocessor.clean_and_engineer(df, one_hot=False)
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
        n_epochs=200,
        batch_size=258,
        lr=3e-4,
        weight_decay=1e-4,
        patience=100,
        hidden_layers=(128, 64, 32),
        verbose=True
    )
    
    print("\n5. Training completed!")
    print(f"Model saved as 'best_intrinsic_mlp.pt'")
    print(f"Feature scaler and names available for predictions")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history["rmse"], label="Training RMSE")
    plt.plot(history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history["val_acc_5pct"], label="< 5% Accuracy", color='green')
    plt.axhline(y=24.60, color='red', linestyle='--', label='Target (WITHOUT Loc)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("5% Error Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history["val_acc_15pct"], label="< 15% Accuracy", color='blue')
    plt.axhline(y=64.54, color='red', linestyle='--', label='Target (WITHOUT Loc)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("15% Error Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history, scaler, feat_names


if __name__ == "__main__":
    model, history, scaler, feat_names = main()
