import ast
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src import features
from src.inference import forecast
from src.prepare_data import split_data
from src.train import train_model, evaluate

ROOT = Path(__file__).resolve().parents[1]


def sample_data():
    return pd.DataFrame([
        {"created_at": day, "increment_id": f"{i}-{j}", "price": float(10+j), "Customer ID": str(j)}
        for i, day in enumerate(pd.date_range("2020-01-01", periods=160))
        for j in range(2+i % 5)
    ])


class PipelineTests(unittest.TestCase):
    def test_exact_feature_transfer(self):
        nb = json.loads((ROOT / "notebooks/4. Модель.ipynb").read_text(encoding="utf-8"))
        original = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell["source"])).body:
                    if isinstance(node, ast.FunctionDef):
                        original[node.name] = ast.dump(node)
        extracted = {n.name: ast.dump(n) for n in ast.parse(
            Path(features.__file__).read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(original, extracted)

    def test_no_same_day_features(self):
        data = sample_data()
        history = features.make_daily_orders(data)
        raw = features.make_daily_raw_features(data)
        day = history.index[50]
        before = features.make_features(history, raw).loc[day].drop("orders")
        history.loc[day:] = 999999
        raw.loc[day:] = 999999
        after = features.make_features(history, raw).loc[day].drop("orders")
        pd.testing.assert_series_equal(before, after)

    def test_round_trip_and_no_future_aggregates(self):
        splits = split_data(sample_data())
        bundle = train_model(splits["train"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(bundle, path)
            result = forecast(path, 7)
            self.assertEqual(len(result), 7)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(result.ge(0).all())
            self.assertEqual(result.index.min(), bundle["history"].index.max() + pd.Timedelta(days=1))
            pd.testing.assert_series_equal(result, forecast(path, 7, splits["train"]))
            with self.assertRaises(ValueError):
                forecast(path, 0)
        metrics, predictions = evaluate(bundle, splits["validation"], splits["test"])
        self.assertTrue(all(np.isfinite(list(m.values())).all() for m in metrics.values()))
        changed = splits["validation"].copy()
        changed["price"] *= 999
        _, other = evaluate(bundle, changed, splits["test"])
        pd.testing.assert_series_equal(predictions["validation"].prediction, other["validation"].prediction)


if __name__ == "__main__":
    unittest.main()
