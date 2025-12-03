# make_pivots_with_pvalues_txt_fixed.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon

RESULTS_DIR = Path("results")
IN_CSV  = RESULTS_DIR / "all_experiments.csv"
OUT_AUC = RESULTS_DIR / "table_auc_with_sd_p.txt"
OUT_ACC = RESULTS_DIR / "table_acc_with_sd_p.txt"

def safe_pvalue_greater(a: pd.Series, b: pd.Series) -> float:
    """
    Return one-sided (greater: a > b) paired p-value.
    Uses Wilcoxon when possible; falls back to paired t-test.
    Returns np.nan if not enough pairs or differences are degenerate.
    """
    a = a.dropna()
    b = b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return np.nan
    a = a.loc[common]
    b = b.loc[common]
    diffs = (a - b).to_numpy()
    # If all diffs are (near) zero or no variance, tests aren't valid
    if np.allclose(diffs, 0) or np.isclose(np.std(diffs, ddof=1), 0):
        return np.nan
    # Try Wilcoxon first
    try:
        _, p = wilcoxon(a, b, zero_method="wilcox", alternative="greater")
        return float(p)
    except Exception:
        # Fall back to paired t-test
        _, p = ttest_rel(a, b, alternative="greater")
        return float(p)

def paired_pvalues(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Paired p-values per dataset comparing each model to the best (by mean of metric).
    """
    rows = []
    for dataset, sub in df.groupby("dataset", sort=False):
        piv = sub.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        # choose a single best model deterministically (idxmax breaks ties by first occurrence)
        best = piv.mean().idxmax()
        best_vals = piv[best]
        for model in piv.columns:
            if model == best:
                p = np.nan
            else:
                p = safe_pvalue_greater(best_vals, piv[model])
            rows.append({"dataset": dataset, "model": model, "best_model": best, "p_value": p})
    return pd.DataFrame(rows)

def make_table(df: pd.DataFrame, metric: str, out_txt: Path):
    # mean & sd across seeds for dataset x model
    stats = (
        df.groupby(["dataset", "model"], as_index=False)
          .agg(mean=(metric, "mean"), sd=(metric, "std"))
    )

    # Determine exactly one best model per dataset (tie-broken deterministically)
    best_by_dataset = (
        stats.sort_values(["dataset", "mean", "model"], ascending=[True, False, True])
             .groupby("dataset", as_index=False)
             .first()[["dataset", "model"]]
             .rename(columns={"model": "best_model"})
    )

    # p-values vs best for this metric
    pvals = paired_pvalues(df, metric)

    # Merge: stats + best + pvals
    merged = (
        stats.merge(best_by_dataset, on="dataset", how="left")
             .merge(pvals[["dataset", "model", "p_value"]], on=["dataset", "model"], how="left")
    )

    # Ensure one row per dataset+model (defensive)
    merged = merged.sort_values(["dataset", "model"]).drop_duplicates(subset=["dataset", "model"], keep="first")

    # Format cells: mean (SD; p=...)
    def fmt(mean, sd, p, is_best):
        m = f"{mean:.2f}" if pd.notna(mean) else "nan"
        s = "nan" if pd.isna(sd) else f"{sd:.2f}"
        if is_best:
            return f"{m} ({s}; —)"
        if pd.isna(p):
            return f"{m} ({s}; p=NA)"
        p_str = f"{p:.1e}" if p < 1e-4 else f"{p:.3f}".rstrip("0").rstrip(".")
        return f"{m} ({s}; p={p_str})"

    merged["is_best"] = merged["model"] == merged["best_model"]
    merged["cell"] = merged.apply(lambda r: fmt(r["mean"], r["sd"], r["p_value"], r["is_best"]), axis=1)

    # Pivot to dataset x model
    table = (
        merged.pivot(index="dataset", columns="model", values="cell")
              .sort_index()
              .sort_index(axis=1)
    )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(table.to_string())
    print(f"Saved {metric.upper()} table -> {out_txt}")

def main():
    df = pd.read_csv(IN_CSV)  # expects: dataset, model, seed, auc, acc
    make_table(df, metric="auc", out_txt=OUT_AUC)
    make_table(df, metric="acc", out_txt=OUT_ACC)

if __name__ == "__main__":
    main()
