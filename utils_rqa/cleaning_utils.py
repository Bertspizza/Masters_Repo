import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import butter, filtfilt

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
        
        x_col = f"{base}_x_offset"
        y_col = f"{base}_y_offset"
        
        if x_col in df.columns and y_col in df.columns:
            mask = df[conf_col] < threshold
            df.loc[mask, x_col] = np.nan
            df.loc[mask, y_col] = np.nan
            
            # Optional: mask raw x/y too if present
            raw_x, raw_y = f"{base}_x", f"{base}_y"
            if raw_x in df.columns and raw_y in df.columns:
                df.loc[mask, raw_x] = np.nan
                df.loc[mask, raw_y] = np.nan
    
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

def butter_lowpass(cutoff, fs, order=4):
    """
    Design a Butterworth lowpass filter.

    Parameters:
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the signal.
    order (int): The order of the filter.

    Returns:
    tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Butterworth filter design
    return b, a

def apply_filter(data, cutoff=10, fs=60, order=4):
    """
    Apply a lowpass Butterworth filter to the data.

    Parameters:
    data (array-like): The input data to be filtered.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the signal.
    order (int): The order of the filter.

    Returns:
    array-like: The filtered data.
    """
    b, a = butter_lowpass(cutoff, fs, order)  # Get filter coefficients
    y = filtfilt(b, a, data)  # Apply the filter to the data
    return y

def filter_data(df):
    """
    Apply a lowpass filter to each numeric column in a DataFrame, ignoring columns with all NaN values.

    Parameters:
    df (pd.DataFrame): The input DataFrame with data to be filtered.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    
    for column in df.columns:
        # Skip columns with all NaN values
        if df[column].isna().all():
            print(f"Skipping column with all NaN values: {column}")
            continue
        
        try:
            # Replace NaNs with 0 for filtering, but preserve original NaNs
            column_data = df[column].fillna(0)
            filtered_data = apply_filter(column_data)
            # Restore original NaNs after filtering
            df[column] = np.where(df[column].isna(), np.nan, filtered_data)
        except Exception as e:
            print(f"Error applying filter to column {column}: {e}")
    
    return df

def center_data(df, columns=None):
    """
    Subtract column means (centering) for numeric columns.
    If columns is None, applies to all numeric columns.
    """
    df_centered = df.copy()
    if columns is None:
        columns = df_centered.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean_val = df_centered[col].mean()
        df_centered[col] = df_centered[col] - mean_val
    
    return df_centered

def zscore_data(df, columns=None):
    """
    Applies z-score normalization: (x - mean) / std.
    """
    df_zscore = df.copy()
    if columns is None:
        columns = df_zscore.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean_val = df_zscore[col].mean()
        std_val = df_zscore[col].std(ddof=0)  # ddof=0 for population std, or ddof=1 for sample std
        if std_val != 0:
            df_zscore[col] = (df_zscore[col] - mean_val) / std_val
        else:
            # If std is zero, all values are the same
            df_zscore[col] = 0
    
    return df_zscore

def unit_interval_data(df, columns=None):
    """
    Scales numeric columns into [0, 1] range:
       (x - min) / (max - min)
    """
    df_minmax = df.copy()
    if columns is None:
        columns = df_minmax.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        min_val = df_minmax[col].min()
        max_val = df_minmax[col].max()
        if max_val != min_val:
            df_minmax[col] = (df_minmax[col] - min_val) / (max_val - min_val)
        else:
            # If all values are the same, scale them to 0
            df_minmax[col] = 0
    
    return df_minmax

def linear_interpolation(df, columns=None):
    """
    Fills missing values in numeric columns using linear interpolation.
    If columns is None, applies to all numeric columns.
    """
    df_interpolated = df.copy()
    if columns is None:
        columns = df_interpolated.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        df_interpolated[col] = df_interpolated[col].interpolate(method='linear', limit_direction='both')
    
    return df_interpolated
    
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

def normalize_data(data, norm):
    if norm == 1:
        return (data - np.min(data)) / (np.max(data) - np.min(data))  # Unit interval
    elif norm == 2:
        return (data - np.mean(data)) / np.std(data)  # Z-score
    elif norm == 3:
        return data - np.mean(data)  # Center around mean
    else:
        print('WARNING: No normalisation was applied. Check parameter settings.')
        return data  # No normalization
    
