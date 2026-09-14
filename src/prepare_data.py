import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean_data(path):
    df = pd.read_csv(path, dtype="string")
    df.columns = df.columns.str.strip()
    empty = [f"Unnamed: {n}" for n in range(21, 26)]
    present = [c for c in empty if c in df]
    if df[present].notna().any().any():
        raise ValueError("Ожидаемые пустые столбцы содержат данные.")
    df = df.drop(columns=present)
    for col in ["item_id", "increment_id", "Customer ID", "sku", "sales_commission_code"]:
        df[col] = df[col].str.strip().replace(r"\N", pd.NA)
    df["MV"] = df["MV"].str.strip().str.replace(",", "", regex=False).replace("-", pd.NA)
    for col in ["price", "qty_ordered", "grand_total", "discount_amount", "MV"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["status", "category_name_1"]:
        df[col] = df[col].replace(r"\N", pd.NA)
    df["BI Status"] = df["BI Status"].replace("#REF!", pd.NA)
    for col in ["created_at", "Working Date", "Customer Since"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Year", "Month"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.dropna(how="all").copy()


def split_data(df):
    if df.groupby("increment_id")["created_at"].nunique().gt(1).any():
        raise ValueError("Один заказ встречается в нескольких датах.")
    orders = (df[["increment_id", "created_at"]].dropna()
              .drop_duplicates("increment_id").sort_values("created_at"))
    if orders.empty:
        raise ValueError("Нет заказов с корректными датами и идентификаторами.")
    cumulative = orders.groupby("created_at").size().sort_index().cumsum()
    train_end = cumulative[cumulative >= len(orders) * 0.70].index[0]
    val_end = cumulative[cumulative >= len(orders) * 0.85].index[0]
    splits = {
        "train": df[df.created_at <= train_end].copy(),
        "validation": df[(df.created_at > train_end) & (df.created_at <= val_end)].copy(),
        "test": df[df.created_at > val_end].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("Одна из выборок пуста; нужно больше дат.")
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    df = clean_data(args.input)
    splits = split_data(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "cleaned_types.parquet", index=False)
    summary = {"source": args.input.name, "cleaned_rows": len(df),
               "excluded_missing_date_rows": int(df.created_at.isna().sum()), "splits": {}}
    for name, part in splits.items():
        part.to_parquet(args.output_dir / f"{name}.parquet", index=False)
        summary["splits"][name] = {"rows": len(part), "orders": int(part.increment_id.nunique()),
                                   "start": str(part.created_at.min()), "end": str(part.created_at.max())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
