import os
import numpy as np
import pandas as pd

DATA_DIR = "./"
NUM_MEMBERS = 199
LAYERS = list(range(7))
WEIGHTS = np.array([0.50, 0.20, 0.10, 0.05, 0.02, 0.02, 0.01])
WEIGHTS = WEIGHTS / WEIGHTS.sum()
FINAL_TIME = 499


def load_final_row(file_path):
    """Return final row as numeric vector with NaNs removed."""
    df = pd.read_csv(file_path)
    row = df[df.iloc[:, 0] == FINAL_TIME].iloc[0, 1:]
    vec = pd.to_numeric(row, errors="coerce").dropna().to_numpy()
    return vec


# -------------------------
# 1. Load all data
# -------------------------
final_values = {m: {} for m in range(1, NUM_MEMBERS + 1)}

for m in range(1, NUM_MEMBERS + 1):
    for layer in LAYERS:
        fpath = os.path.join(DATA_DIR, f"ensemble_{m}_{layer}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")
        final_values[m][layer] = load_final_row(fpath)

# -------------------------
# 2. Determine common vector length
# -------------------------
min_len = min(
    len(final_values[m][layer])
    for m in range(1, NUM_MEMBERS + 1)
    for layer in LAYERS
)

print(f"Detected minimum vector length across all files: {min_len}")

# Trim vectors
for m in range(1, NUM_MEMBERS + 1):
    for layer in LAYERS:
        final_values[m][layer] = final_values[m][layer][:min_len]

# -------------------------
# 3. Compute mean, 5th, 9th percentile
# -------------------------
layer_stats = {}

for layer in LAYERS:
    stack = np.stack([final_values[m][layer] for m in range(1, NUM_MEMBERS + 1)])
    layer_stats[layer] = {
        "mean": np.mean(stack, axis=0),
        "p05": np.percentile(stack, 5, axis=0),
        "p95": np.percentile(stack, 95, axis=0),
    }

# -------------------------
# 4. Compute weighted distances
# -------------------------
dist_mean = {}
dist_p05 = {}
dist_p95 = {}

for m in range(1, NUM_MEMBERS + 1):

    w_mean = w_p05 = w_p95 = 0.0

    for i, layer in enumerate(LAYERS):
        vals = final_values[m][layer]
        mean_v = layer_stats[layer]["mean"]
        p05_v = layer_stats[layer]["p05"]
        p95_v = layer_stats[layer]["p95"]

        w = WEIGHTS[i]

        w_mean += w * np.linalg.norm(vals - mean_v)
        w_p05 += w * np.linalg.norm(vals - p05_v)
        w_p95 += w * np.linalg.norm(vals - p95_v)

    dist_mean[m] = w_mean
    dist_p05[m] = w_p05
    dist_p95[m] = w_p95

# -------------------------
# 5. Find closest ensemble members
# -------------------------
best_mean = min(dist_mean, key=dist_mean.get)
best_p05  = min(dist_p05,  key=dist_p05.get)
best_p95  = min(dist_p95,  key=dist_p95.get)

print("\n==============================")
print(" BEST MATCHING ENSEMBLE MEMBERS")
print("==============================")
print(f"Closest to MEAN:       {best_mean}   (dist = {dist_mean[best_mean]:.6f})")
print(f"Closest to 5th pct:    {best_p05}    (dist = {dist_p05[best_p05]:.6f})")
print(f"Closest to 95th pct:    {best_p95}    (dist = {dist_p95[best_p95]:.6f})")
print("==============================")
