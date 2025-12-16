import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, sosfiltfilt

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def get_window_indices(data_length, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return [(start, start + window_size) for start in range(0, data_length - window_size + 1, step)]

def mask_low_confidence(df, threshold=0.3):
    """
    Replace keypoint coordinates with NaN if confidence < threshold.
    Works with columns ending in '_confidence'.
    """
    df = df.copy()
    conf_cols = [c for c in df.columns if c.endswith("_confidence")]
    
    for conf_col in conf_cols:
        base = conf_col.rsplit("_", 1)[0]  # remove "_confidence"
        
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        
        if x_col in df.columns and y_col in df.columns:
            mask = df[conf_col] < threshold
            df.loc[mask, x_col] = np.nan
            df.loc[mask, y_col] = np.nan
    
    return df

def interpolate_nans(data, max_gap=60): 
    """
    Linearly interpolate NaN values in a DataFrame or Series, 
    but only for gaps <= max_gap. Longer gaps remain NaN.

    Parameters:
    - data (pd.DataFrame or pd.Series): The data with potential NaN values.
    - max_gap (int): Maximum number of consecutive NaNs to fill.

    Returns:
    - pd.DataFrame or pd.Series: Data with NaN values linearly interpolated 
      where gaps <= max_gap.
    """
    if isinstance(data, pd.DataFrame):
        return data.interpolate(
            method='linear', 
            axis=0, 
            limit=max_gap, 
            limit_direction='both'
        )
    elif isinstance(data, pd.Series):
        return data.interpolate(
            method='linear', 
            limit=max_gap, 
            limit_direction='both'
        )
    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")
    
def norm_by_res(df, width=720, height=1152): 
    df_norm = df.copy()

    # Normalise x values
    x_cols = [col for col in df.columns if "_x" in col or "_x_offset" in col]
    df_norm[x_cols] = df_norm[x_cols] / width
 
    # Normalise y values
    y_cols = [col for col in df.columns if "_y" in col or "_y_offset" in col]
    df_norm[y_cols] = df_norm[y_cols] / height
 
    return df_norm

def trim_start_frame(data, file_name, meta_data_path):
    """
    Trims the initial rows from a pose DataFrame based on start_frame info
    in a metaData_coding.csv file.

    Parameters:
    - data (pd.DataFrame): The pose data to trim.
    - file_name (str): The filename of the current pose file (e.g., 'p1_001_coding_1_pose.csv').
    - meta_data_path (str): Path to the metaData_coding.csv file.

    Returns:
    - pd.DataFrame: Trimmed pose data.
    """
    # Extract couple and condition from filename
    try:
        base = file_name.split('/')[-1].split('\\')[-1]  # Support both Windows & Linux paths
        parts = base.split('_')
        couple = int(parts[1])
        condition = parts[4]
    except Exception as e:
        print(f"[ERROR] Could not parse filename: {file_name} — {e}")
        return data

    # Load metadata and find matching row
    meta_data = pd.read_csv(meta_data_path)
    matching_row = meta_data[(meta_data['couple'] == couple) & (meta_data['trial'] == condition)]

    if not matching_row.empty:
        start_frame = int(matching_row.iloc[0]['start_frame'])
        data = data.iloc[start_frame:].reset_index(drop=True)
    else:
        print(f"[WARN] No matching metadata for {file_name} (Couple: {couple}, Condition: {condition})")
    
    if couple == 165 and condition == 'trial1': # Video continued after the conversation was concluded. 
        data = data.iloc[:11785]
    return data

def linear_detrend(df, columns=None):
    """
    Removes linear trends from numeric columns by subtracting the best-fit line.
    If columns is None, applies to all numeric columns.
    """

    df_detrended = df.copy()
    if columns is None:
        columns = df_detrended.select_dtypes(include=[np.number]).columns

    for col in columns:
        x = np.arange(len(df_detrended[col]))
        slope, intercept, _, _, _ = linregress(x, df_detrended[col])
        df_detrended[col] = df_detrended[col] - (slope * x + intercept)

    return df_detrended

def filter_data(df, cutoff=10, fs=25, order=2):
    """
    Apply a lowpass Butterworth filter to each numeric column in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input data.
    cutoff (float): Cutoff frequency in Hz.
    fs (float): Sampling frequency in Hz.
    order (int): Filter order.

    Returns:
    pd.DataFrame: Filtered data.
    """
    b, a = butter(order, cutoff, btype='low', fs=fs) 

    df = df.copy()
    for col in df.select_dtypes(include=[float, int]).columns:
        df[col] = filtfilt(b, a, df[col])

    return df

def filter_data_safe_preserve_nans(
    df: pd.DataFrame,
    fs: float = 25.0,
    cutoff: float = 8.0,
    order: int = 4,
    audit: bool = True,
) -> pd.DataFrame:
    """
    Low‑pass each numeric column with zero‑phase Butterworth (sosfiltfilt),
    filling gaps only internally for stability, then RESTORING original NaNs.
    - If the series is too short or filtering fails, returns the original column (NaNs intact).
    """
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not num_cols:
        return out

    # Butterworth design
    wn = cutoff / (fs / 2.0)
    wn = min(max(wn, 1e-6), 0.999999)  # clamp to (0,1)
    sos = butter(order, wn, btype='low', output='sos')

    all_nan_before, too_short, pad_failed = [], [], []

    for c in num_cols:
        x = out[c].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(x)

        if not finite.any():
            all_nan_before.append(c)
            continue

        # Internal fill for filter stability (linear across gaps + hold edges)
        idx = np.arange(x.size)
        x_filled = x.copy()
        if not finite.all():
            x_filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
            first, last = np.where(finite)[0][[0, -1]]
            x_filled[:first] = x_filled[first]
            x_filled[last+1:] = x_filled[last]

        # Heuristic minimum length for filtfilt
        min_len = max(25, 4 * order + 5)
        if x_filled.size < min_len:
            too_short.append(c)
            # Keep original (but ensure original NaNs remain)
            out[c] = x
            continue

        # Filter; if it fails (padding issues), keep original
        try:
            y = sosfiltfilt(sos, x_filled, padtype='odd')
        except Exception:
            pad_failed.append(c)
            out[c] = x
            continue

        # Restore genuine missing samples
        y[~finite] = np.nan
        out[c] = y

    if audit:
        if all_nan_before:
            print(f"[filter_data_safe_preserve_nans] All‑NaN (unchanged): {len(all_nan_before)} cols (e.g., {all_nan_before[:5]})")
        if too_short:
            print(f"[filter_data_safe_preserve_nans] Unfiltered (too short): {len(too_short)} cols (e.g., {too_short[:5]})")
        if pad_failed:
            print(f"[filter_data_safe_preserve_nans] Unfiltered (pad failure): {len(pad_failed)} cols (e.g., {pad_failed[:5]})")

    return out

