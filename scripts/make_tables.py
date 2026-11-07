#!/usr/bin/env python
"""Build nDCG@10 (%) tables from results.csv files.

    python scripts/make_tables.py results/beir_main/results.csv results/beir_baselines/results.csv --best-of
    python scripts/make_tables.py results/dapr/results.csv --level chunk --delta fixed@480
"""

import argparse
import re

import pandas as pd

FAMILIES = {
    r"^fixed_chars\d+$": "Fixed (best)",
    r"^recursive_chars\d+$": "Recursive (best)",
    r"^langchain_": "LangChain",
}


def family(method: str) -> str:
    for pattern, name in FAMILIES.items():
        if re.match(pattern, method):
            return name
    return method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--level", default="doc", choices=["doc", "chunk"])
    parser.add_argument("--encoder", help="keep only this encoder (BM25 rows are always kept)")
    parser.add_argument("--best-of", action="store_true", help="keep the best configuration of each swept baseline")
    parser.add_argument("--delta", metavar="REFERENCE", help="add a row: each method minus REFERENCE")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df = df[df["level"] == args.level]
    if args.encoder:
        df = df[(df["encoder"] == args.encoder) | (df["encoder"] == "-")]
    df = df.assign(value=100.0 * df[args.metric].astype(float))
    if args.best_of:
        df = df.assign(method=df["method"].map(family))

    table = df.pivot_table(index="method", columns="subset", values="value", aggfunc="max")
    table = table.reindex(columns=list(dict.fromkeys(df["subset"])))
    if args.delta:
        if args.delta not in table.index:
            raise SystemExit(f"--delta: no method {args.delta!r} in {list(table.index)}")
        for method in [m for m in table.index if m != args.delta and not m.startswith("Δ")]:
            table.loc[f"Δ {method}"] = table.loc[method] - table.loc[args.delta]

    print(table.round(2).to_latex(float_format="%.2f") if args.latex else table.round(2).to_markdown())


if __name__ == "__main__":
    main()
