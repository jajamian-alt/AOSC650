import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from scipy.ndimage import maximum_filter1d

with open('runs_output.jsonl', 'r') as file:
    df = pd.read_json(file, lines=True)

df_expanded = pd.concat([df.drop(columns=['keplerianElements']), df['keplerianElements'].apply(pd.Series)], axis=1)
df_expanded = df_expanded.drop(columns = ["comments", "anomalyType", "mu"])
df_expanded["residTimeSeries"] = df_expanded["residTimeSeries"].apply(np.array)

time_series = df_expanded["residTimeSeries"].apply(np.array)

num = 8

plt.scatter(time_series[num][:, 0], time_series[num][:, 1], marker = 'o', label = "Sampled Residuals")
plt.axvline(df_expanded["maneuverEpoch"][num], color='r', label = "Maneuver Epoch")
plt.xlabel("Epoch (Seconds)")
plt.ylabel("Residual Magnitude (Sigma)")
plt.title("Residual vs. Time")
plt.legend()
plt.show()
#
# def fit_row_with_residuals(matrix):
#     x = matrix[:, 0]
#     y = matrix[:, 1]
#
#     # Linear fit y = m*x + b
#     m, b = np.polyfit(x, y, 1)
#
#     # Compute residuals
#     residuals = y - (m * x + b)
#
#     # Create Nx2 residuals matrix: first column x, second column residual
#     resid_matrix = np.column_stack((x, residuals))
#
#     return pd.Series([m, b, resid_matrix])
#
# # Apply function row-wise
# df_expanded[['linear_fit_slope', 'linear_fit_intercept', 'linear_fit_residuals']] = df_expanded["residTimeSeries"].apply(fit_row_with_residuals)
#
# # We want to pad to length max_len by repeating the first row
# def padTimeSeriesColumn(timeSeries):
#     max_len = max(ts.shape[0] for ts in timeSeries)
#
#     padded_ts = []
#
#     for ts in timeSeries:
#         L = ts.shape[0]
#         pad_len = max_len - L
#         if pad_len > 0:
#             # repeat the first row pad_len times
#             pad = np.repeat(ts[0:1, :], pad_len, axis=0)
#             ts_padded = np.vstack((pad, ts))  # prepend padding
#         else:
#             ts_padded = ts
#         padded_ts.append(ts_padded)
#
#     return np.stack(padded_ts, axis=0)  # shape: (num_samples, max_len, 2)
#
def interpolate_timeseries(ts_series: pd.Series, target_len: int, normalize : bool = True) -> np.ndarray:
    """
    Linearly interpolate a pandas Series of Nx2 time series arrays to a fixed length.

    Parameters:
        ts_series (pd.Series): Series where each element is an (Ni,2) NumPy array
        target_len (int): Desired length to interpolate to
        normalize (bool) sets time width to be 0 - 1 automatically

    Returns:
        np.ndarray: 3D array of shape (num_samples, target_len, 2)
        list(double): observation_window_width
        list(double): observation_window_offset
    """
    all_ts_resampled = []
    observation_window_width = []
    observation_window_offset = []

    for ts in ts_series:
        N = ts.shape[0]
        time = ts[:, 0]
        value = ts[:, 1]
        time_range = time[-1] - time[0]

        if N == target_len:
            ts_resampled = ts
        else:
            # Normalize original indices to [0,1]
            old_idx = (time - time[0]) / time_range
            new_idx = np.linspace(0, 1, target_len)
            # Interpolate both columns
            ts_resampled = np.stack([
                np.interp(new_idx, old_idx, time),  # time column
                np.interp(new_idx, old_idx, value)   # residual column
            ], axis=1)
            if normalize:
               ts_resampled[:,0] = new_idx
        all_ts_resampled.append(ts_resampled)
        observation_window_width.append(time_range)
        observation_window_offset.append(time[0])

    # Stack into 3D array
    return np.stack(all_ts_resampled, axis=0), observation_window_width, observation_window_offset


# interp_value, _, _ = interpolate_timeseries(df_expanded["linear_fit_residuals"], 1000)
# np.save("linear_fit_residuals_sampled.npy", interp_value)
#
# interp_value, _, _ = interpolate_timeseries(df_expanded["residTimeSeries"], 1000)
# np.save("residuals_sampled.npy", interp_value)
#
interp_value, time_width, time_offset = interpolate_timeseries(abs(df_expanded["residTimeSeries"]), 1000)
np.save("residuals_sampled_abs.npy", interp_value)

df_expanded["time_series_offset"] = time_offset
df_expanded["time_series_width"] = time_width
df_expanded["maneuver_scaled_epoch"] = (df_expanded["maneuverEpoch"] - df_expanded["time_series_offset"]) / df_expanded["time_series_width"]
#
# df_expanded = df_expanded.drop(columns = ["linear_fit_residuals", "residTimeSeries"])
#
df_expanded.to_pickle('cleaned_data.pkl')

def smooth_max_window(arr, window=5):
    return maximum_filter1d(arr, size=window, axis=1, mode='nearest')

interp_smoothed = smooth_max_window(interp_value, window=10)
np.save("interp_smoothed.npy", interp_smoothed)

# Scale to be within 0 and 1
# Compute min/max over time for column 1
col = interp_smoothed[:, :, 1]  # shape (n, 1000)

col_min = col.min(axis=1, keepdims=True)  # (n, 1)
col_max = col.max(axis=1, keepdims=True)  # (n, 1)

# Scale only column 1
interp_smoothed[:, :, 1] = (col - col_min) / (col_max - col_min + 1e-8)
np.save("interp_smoothed_scaled.npy", interp_smoothed)








