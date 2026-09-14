"""Рекурсивный прогноз из сохранённой модели, без фактических будущих данных."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .features import make_daily_orders, make_daily_raw_features, recursive_forecast

ROOT = Path(__file__).resolve().parents[1]


def forecast(model_path, days=7, observed=None):
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise ValueError("Горизонт должен быть положительным целым числом.")
    bundle = joblib.load(model_path)
    if bundle.get("format_version") != 1:
        raise ValueError("Неизвестный формат артефакта модели.")
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError(f"Нужен scikit-learn=={bundle['sklearn_version']}.")
    history, raw = bundle["history"], bundle["raw_daily_features"]
    if observed is not None:
        observed = observed.copy()
        observed["created_at"] = pd.to_datetime(observed["created_at"])
        if observed.empty or observed.created_at.isna().any():
            raise ValueError("История должна содержать строки с корректными датами.")
        history = make_daily_orders(observed)
        raw = make_daily_raw_features(observed)
        if len(history) < 29:
            raise ValueError("Передайте полную наблюдаемую историю минимум за 29 дней.")
        if history.index.max() < pd.Timestamp(bundle["training_end"]):
            raise ValueError("История не может заканчиваться раньше обучающей выборки.")
        missing = [c[4:] for c in bundle["feature_columns"] if c.startswith("raw_") and c[4:] not in raw]
        if missing:
            raise ValueError(f"В истории отсутствуют исходные признаки или нарушены их типы: {missing}")
    dates = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=days, freq="D")
    return recursive_forecast(bundle["model"], history, dates,
                              bundle["feature_columns"], raw).rename_axis("date")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/model.joblib")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--history", type=Path, help="Полная очищенная наблюдаемая история, Parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/predictions/forecast.csv")
    args = parser.parse_args()
    observed = pd.read_parquet(args.history) if args.history else None
    result = forecast(args.model, args.days, observed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(result.to_string())


if __name__ == "__main__":
    main()
