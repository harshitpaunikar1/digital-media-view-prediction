"""
Digital media view count prediction pipeline.
Predicts article/video view counts from content metadata and engagement signals.
Compares regression and gradient boosting approaches with log-transformed targets.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class MediaViewPredictor:
    """
    Predicts view counts for digital media content (articles, videos, social posts).
    Applies log1p transformation to stabilize the target distribution.
    Provides RMSLE, MAE, and R2 metrics for model comparison.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "views", log_transform: bool = True):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.log_transform = log_transform
        self.models: Dict[str, object] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _build_preprocessor(self):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat",
                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _build_estimators(self) -> Dict:
        estimators = {
            "Ridge": Ridge(alpha=10.0),
            "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                           max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            estimators["XGBoost"] = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                                      max_depth=5, random_state=42,
                                                      tree_method="hist", verbosity=0)
        return estimators

    def rmsle(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Log Error for count prediction."""
        actual = np.maximum(actual, 0)
        predicted = np.maximum(predicted, 0)
        return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))

    def fit_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        """Train all models and return comparison DataFrame sorted by RMSLE."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        feature_cols = self.numeric_features + self.categorical_features
        df = df[feature_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        X = df[feature_cols]
        y_raw = df[self.target_col].values
        y = np.log1p(y_raw) if self.log_transform else y_raw

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        y_test_raw = np.expm1(y_test) if self.log_transform else y_test

        preprocessor = self._build_preprocessor()
        estimators = self._build_estimators()
        self.results = []

        for name, est in estimators.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
            pipe.fit(X_train, y_train)
            preds_log = pipe.predict(X_test)
            preds_raw = np.expm1(preds_log) if self.log_transform else preds_log
            preds_raw = np.maximum(preds_raw, 0)
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "rmsle": round(self.rmsle(y_test_raw, preds_raw), 4),
                "mae": round(float(mean_absolute_error(y_test_raw, preds_raw)), 1),
                "r2": round(float(r2_score(y_test, preds_log)), 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("rmsle").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict view counts (raw scale) for new data."""
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit_and_evaluate first.")
        feature_cols = self.numeric_features + self.categorical_features
        preds = self.models[self.best_model_name].predict(df[feature_cols])
        return np.expm1(preds).astype(int) if self.log_transform else preds.astype(int)

    def viral_probability(self, predicted_views: np.ndarray,
                          viral_threshold: int = 100_000) -> np.ndarray:
        """Return binary array indicating whether predicted views exceed viral threshold."""
        return (predicted_views >= viral_threshold).astype(int)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        feature_names = self.numeric_features + cat_names
        return pd.DataFrame({
            "feature": feature_names[:len(est.feature_importances_)],
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame({
        "word_count": np.random.randint(200, 5000, n),
        "images_count": np.random.randint(0, 20, n),
        "publish_hour": np.random.randint(0, 24, n),
        "num_shares": np.random.randint(0, 500, n),
        "num_comments": np.random.randint(0, 200, n),
        "avg_sentence_length": np.random.uniform(8, 25, n).round(1),
        "content_type": np.random.choice(["article", "video", "infographic", "listicle"], n),
        "category": np.random.choice(["tech", "politics", "sports", "entertainment", "health"], n),
        "views": np.abs(np.random.lognormal(mean=8, sigma=2, size=n)).astype(int),
    })

    predictor = MediaViewPredictor(
        numeric_features=["word_count", "images_count", "publish_hour",
                          "num_shares", "num_comments", "avg_sentence_length"],
        categorical_features=["content_type", "category"],
    )
    results = predictor.fit_and_evaluate(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {predictor.best_model_name}")

    sample_preds = predictor.predict(df.head(10))
    viral = predictor.viral_probability(sample_preds)
    for pred, v in zip(sample_preds[:5], viral[:5]):
        print(f"  Predicted views: {pred:,} | Viral: {bool(v)}")

    imp = predictor.feature_importance()
    if imp is not None:
        print("\nTop 5 features:")
        print(imp.head(5).to_string(index=False))
