import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import make_daily_orders, make_daily_raw_features, make_features, recursive_forecast

ROOT = Path(__file__).resolve().parents[1]


def train_model(train):
    train = train.copy()
    train["created_at"] = pd.to_datetime(train["created_at"])
    if train.empty or train.created_at.isna().any():
        raise ValueError("Нужна непустая обучающая выборка с корректными датами.")
    history = make_daily_orders(train)
    if len(history) < 29:
        raise ValueError("Для лагов и окон необходима история минимум за 29 дней.")
    raw = make_daily_raw_features(train)
    features = make_features(history, raw).dropna(axis=1, how="all")
    X = features.drop(columns="orders")
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.03, max_iter=500, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=5, random_state=42,
    ).fit(X, features["orders"])
    return {"format_version": 1, "model": model, "feature_columns": list(X.columns),
            "history": history, "raw_daily_features": raw,
            "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
            "numpy_version": np.__version__, "training_end": str(history.index.max().date())}


def evaluate(bundle, validation, test):
    """Fixed-origin evaluation: no actual aggregates inside each forecast horizon."""
    history = bundle["history"]
    raw = bundle["raw_daily_features"]
    metrics, predictions = {}, {}
    for name, frame in [("validation", validation), ("test", test)]:
        frame = frame.copy()
        frame["created_at"] = pd.to_datetime(frame["created_at"])
        actual = make_daily_orders(frame)
        if actual.index.min() <= history.index.max():
            raise ValueError("Периоды должны идти строго последовательно без пересечений.")
        dates = pd.date_range(history.index.max() + pd.Timedelta(days=1), actual.index.max())
        pred = recursive_forecast(bundle["model"], history, dates,
                                  bundle["feature_columns"], raw).reindex(actual.index)
        metrics[name] = {"MAE": float(mean_absolute_error(actual, pred)),
                         "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                         "R2": float(r2_score(actual, pred))}
        predictions[name] = pd.DataFrame({"actual": actual, "prediction": pred}).rename_axis("date")
        # Validation becomes available before the test forecast origin; no refit.
        history = pd.concat([history, actual]).asfreq("D", fill_value=0)
        raw = pd.concat([raw, make_daily_raw_features(frame)])
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model", type=Path, default=ROOT / "models/model.joblib")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/metrics")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    source = args.data_dir / "train.parquet"
    bundle = train_model(pd.read_parquet(source))
    bundle["training_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    if args.evaluate:
        metrics, _ = evaluate(bundle, pd.read_parquet(args.data_dir / "validation.parquet"),
                                        pd.read_parquet(args.data_dir / "test.parquet"))
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        report = {"protocol": "fixed_origin_no_future_raw_aggregates", "metrics": metrics}
        (args.reports_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved trained model and inference context: {args.model}")


if __name__ == "__main__":
    main()
