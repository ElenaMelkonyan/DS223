import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

with open("data/bass_fit_params.json", "r") as f:
    params = json.load(f)

p_est = params["p"]
q_est = params["q"]
M_est = params["M"]

# Here we assume a smaller market potential, e.g. 10 million units.
M_target = 10_000_000

def bass_adopters(t, p, q, M):
    num = (p + q)**2 * np.exp(-(p + q) * t)
    den = p * (1 + (q/p) * np.exp(-(p + q) * t))**2
    return M * (num / den)

# === Forecast next 10 years ===
years_forecast = np.arange(1, 11)
pred_new = bass_adopters(years_forecast, p_est, q_est, M_target)
pred_cum = np.cumsum(pred_new)

forecast_df = pd.DataFrame({
    "Year_since_launch": years_forecast,
    "Pred_new_adopters": np.round(pred_new).astype(int),
    "Cumulative_adopters": np.round(pred_cum).astype(int)
})

# === Save results ===
forecast_df.to_csv("data/beamo_forecast.csv", index=False)

print("=== Forecast – Withings BeamO (Global) ===")
print(f"Using parameters from Fitbit analog:\np={p_est:.4f}, q={q_est:.4f}, M_target={M_target:,.0f}\n")
print(forecast_df)

# === Plot forecast ===
plt.figure(figsize=(7,4))
plt.plot(
    forecast_df["Year_since_launch"],
    forecast_df["Pred_new_adopters"],
    "o-",
    label="Predicted new adopters"
)
plt.title("Withings BeamO – Forecasted Diffusion (Global)")
plt.xlabel("Years since launch")
plt.ylabel("New adopters (units)")
plt.legend()
plt.tight_layout()
plt.savefig("img/beamo_forecast.png", dpi=200)
plt.close()
