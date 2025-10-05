import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Devices_Sold_Units" in df.columns:
        shipments = df["Devices_Sold_Units"].astype(float).values
    elif "Devices_Sold_Millions" in df.columns:
        shipments = (df["Devices_Sold_Millions"].astype(float).values * 1_000_000)
    else:
        # fallback: any shipments-like column
        cand = [c for c in df.columns if c.lower().startswith("ship")]
        if not cand:
            raise ValueError("CSV must have Devices_Sold_Units or Devices_Sold_Millions (or a shipments column).")
        shipments = df[cand[0]].astype(float).values
    years = df["Year"].astype(int).values
    order = np.argsort(years)
    return pd.DataFrame({"year": years[order], "sales": shipments[order]})

def bass_adopters(t, p, q, M):
    t = np.asarray(t, dtype=float)
    num = (p + q)**2 * np.exp(-(p + q) * t)
    den = p * (1 + (q/p) * np.exp(-(p + q) * t))**2
    return M * (num / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/Dataset.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    df = load_series(args.csv).reset_index(drop=True)
    y = df["sales"].values
    t = np.arange(1, len(y) + 1, dtype=float)

    # Initial guesses & bounds (keep p, q positive; M at least bigger than max yearly sales)
    p0, q0 = 0.03, 0.5
    M0 = float(y.sum() * 1.5)
    bounds = ([1e-6, 1e-6, float(max(y))], [1.0, 3.0, 1e12])

    popt, pcov = curve_fit(
        bass_adopters, t, y, p0=[p0, q0, M0], bounds=bounds, maxfev=200000
    )
    p_est, q_est, M_est = [float(v) for v in popt]

    # Peak time t* = ln(q/p) / (p+q) (only if q>p>0)
    t_peak = float(np.log(q_est / p_est) / (p_est + q_est)) if (p_est > 0 and q_est > 0) else float("nan")

    # Predictions, R^2
    yhat = bass_adopters(t, p_est, q_est, M_est)
    sst = float(((y - y.mean())**2).sum())
    sse = float(((y - yhat)**2).sum())
    r2 = float(1 - sse/sst) if sst > 0 else float("nan")

    # Save results table
    out = df.copy()
    out["pred_sales"] = yhat
    out["cum_sales"] = out["sales"].cumsum()
    out["cum_pred"]  = out["pred_sales"].cumsum()
    csv_path = f"{args.outdir.rstrip('/')}/bass_fit_results.csv"
    out.to_csv(csv_path, index=False)

    # Save params
    params = {"p": p_est, "q": q_est, "M": M_est, "t_peak": t_peak, "R2": r2}
    json_path = f"{args.outdir.rstrip('/')}/bass_fit_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    # Plot
    plt.figure(figsize=(8,4.5))
    plt.bar(out["year"], out["sales"], label="Actual", alpha=0.7)
    plt.plot(out["year"], out["pred_sales"], "r-o", label="Bass fit")
    plt.title("Fitbit Annual Devices Sold vs. Bass Model Fit")
    plt.xlabel("Year"); plt.ylabel("Units per year")
    plt.legend()
    plt.tight_layout()
    plot_path = f"{args.outdir.rstrip('/')}/bass_fit_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Console summary
    print("=== Bass Model Results (Fitbit look-alike) ===")
    print(json.dumps(params, indent=2))
    print("Saved:")
    print(" ", json_path)
    print(" ", csv_path)
    print(" ", plot_path)

if __name__ == "__main__":
    main()
