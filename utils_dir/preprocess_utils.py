import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, sosfiltfilt
from utils_dir.align import *
from tqdm import tqdm

def norm_by_res(df, width=720, height=1152): 
    """
    Normalises the x and y coordinates in the dataframe by the given width and height.
    Args:
        df (pd.DataFrame): DataFrame containing keypoint data with x and y coordinates.
        width (int): Width of the pose resolution.
        height (int): Height of the pose resolution.
    Returns:
        pd.DataFrame: DataFrame with normalised x and y coordinates.
    """
    df_norm = df.copy()
    # Normalise x values
    x_cols = [col for col in df.columns if "_x" in col or "_x_offset" in col]
    df_norm[x_cols] = df_norm[x_cols] / width
    # Normalise y values
    y_cols = [col for col in df.columns if "_y" in col or "_y_offset" in col]
    df_norm[y_cols] = df_norm[y_cols] / height
 
    return df_norm

    
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

def read_and_filter(file_path, 
                    columns_to_keep, 
                    output_data_path, 
                    conf_threshold=None, 
                    interpolate_max=25,
                    meta_data=None):
    """
    Reads a CSV file, selects specific columns, performs linear interpolation on missing values,
    and counts the number of missing values per row.

    Args:
        file_path (str): Path to the CSV file.
        columns_to_keep (list): List of column names to retain from the CSV.
        conf_threshold (float): Minimum confidence value to retain offset values. Defaults to 0.
        interpolate_max(int): Maxium sequence of missing values to interpolate. 


    Returns:
        dict: A dictionary with two entries:
              - 'data': The filtered and interpolated DataFrame.
              - 'missing_info': A DataFrame summarizing missing values per row.
    """
    # Extract couple and condition from file name
    file_name = Path(file_path).stem
    couple = int(file_name.split('_')[1])
    condition = file_name.split('_')[4]

    #  Create a set of columns to read based on the keypoints and suffixes
    suffixes = ['_x', '_y', '_confidence']
    columns = {f"{kp}{suffix}" for kp in columns_to_keep for suffix in suffixes}
    
    # Read CSV and retain only specified columns
    data = pd.read_csv(file_path, usecols=lambda col: col in columns)

    data = data.copy()

    # Trim data based on meta data start frame
    if meta_data is not None:
        meta_data = pd.read_csv(meta_data)
        matching_row = meta_data[(meta_data['couple'] == couple) & (meta_data['trial'] == condition)]

        if not matching_row.empty:
            start_frame = int(matching_row.iloc[0]['start_frame'])
            data = data.iloc[start_frame:].reset_index(drop=True)
        else:
            print(f"Cannot find matching meta data for {file_name}")
        
        if couple == 165 and condition == 'trial1': # Video continued after the conversation had concluded. 
            data = data.iloc[:11785]        
    
    # Normalise the Data by pose resolution
    data = norm_by_res(data, width=720, height=1152)

    # Create confidence columns
    conf_cols = [col for col in data.columns if col.endswith("_confidence")]
    # Set x,y to NaN where confidence is below threshold or both x and y are zero
    for conf_col in conf_cols:
        obs = conf_col.replace("_confidence", "")
        x_col = f"{obs}_x"
        y_col = f"{obs}_y"
        # Only proceed if both x and y columns exist
        if x_col in data.columns and y_col in data.columns:
            if conf_threshold is not None:
                conf = data[conf_col]
                data.loc[conf < conf_threshold, [x_col, y_col]] = np.nan
            else:
                data.loc[(data[x_col] == 0) & (data[y_col] == 0), [x_col, y_col]] = np.nan

    # Linear interpolation function for gaps ≤ 30
    def interpolate_or_blank(column):
        return column.interpolate(method='linear', axis=0, limit=interpolate_max, limit_direction='both')

    # Apply interpolation to each column
    data = data.apply(interpolate_or_blank)

    # Apply butterworth filter
    data = filter_data_safe_preserve_nans(data, fs=25, cutoff=10, order=4, audit=False) ############

    # Count missing values per col
    missing_per_col = data.isna().sum()

    # Create dataframe with file name and missing count
    missing_info = pd.DataFrame({
        'file_name': file_name,
        'column': missing_per_col.index,
        'missing_per_column': missing_per_col.values
    })

    # Save filtered data to CSV if output path is provided
    if output_data_path is not None:
        data.to_csv(f"{output_data_path}/{file_name}.csv", index=False)

    return {
        'data': data,
        'missing_info': missing_info
    }


def process_all_csv_files(directory, columns_to_keep, filtered_data_path=None,
                          conf_threshold=None, interpolate_max=None, meta_data=None, rem_couples=[]):
    """
    Processes all CSV files in a given directory: reads and filters each file, interpolates missing values,
    and aggregates the results.

    Args:
        directory (str): Path to the directory containing CSV files.
        columns_to_keep (list): Columns to retain from each CSV.
        output_data_path (str): Path that stores output data after being filtered
        output_missing_path (str): Path that store values missing per column
        filter_conf (bool): Remove values where conf below threshol
        conf_threshold (flt): Confidence threshold for removing values
        interpolate_max (int): Max sequence of missing data to interpolate

    Returns:
        dict: A dictionary containing:
              - 'processed_data': A dict of DataFrames (one per file).
              - 'missing_info_all_files': A single DataFrame summarizing missing values across files.
    """
    # List all CSVs in the directory
    csv_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.csv') and f.split('_')[1] not in rem_couples
    ] 
    
    # Process each file and collect results
    results = [read_and_filter(file_path=file, 
                               columns_to_keep=columns_to_keep,
                               output_data_path=filtered_data_path, 
                               conf_threshold=conf_threshold, 
                               interpolate_max=interpolate_max,
                               meta_data=meta_data) for file in tqdm(csv_files, desc='Filtering OP csv files')]

    # Combine all missing_info into one dataframe
    all_missing_info = pd.concat([res['missing_info'] for res in results], ignore_index=True)

    # Store all processed dataframes in a dictionary
    all_processed_data = {
        res['missing_info']['file_name'].iloc[0]: res['data']
        for res in results
    }

    return {
        'processed_data': all_processed_data,
        'missing_info_all_files': all_missing_info
    }

def calculate_metrics(data, columns):
    """
    Calculates motion and blink-related metrics from facial keypoints.

    Args:
        data (DataFrame): A single pre-processed data frame.
        keyframes_x, keyframes_y (list): Columns used to compute facial center metrics.
        keyframes_top_left_eye, keyframes_bottom_left_eye (list): Columns for left eye blink.
        keyframes_top_right_eye, keyframes_bottom_right_eye (list): Columns for right eye blink.

    Returns:
        dict: A dictionary of series, each representing a calculated metric.
    """
    
    # Drop columns in the dataframe that arent in the expected keypoint arrays
    def drop_keypoints(expected_columns, actual_columns):
        return np.array([k for k in expected_columns if k in actual_columns])
    
    delta_t = 1/25

    # ----------------------------- Calculate head movement -----------------------------
    head_x = data[['Nose_x', 'REye_x', 'LEye_x']].mean(axis=1)
    head_y = data[['Nose_y', 'REye_y', 'LEye_y']].mean(axis=1)

    head_ed_pos = np.sqrt(head_x**2 + head_y**2)
    head_ed_vel = np.sqrt(np.diff(head_x, prepend=np.nan)**2 + 
                                np.diff(head_y, prepend=np.nan)**2)
    
    head_ed_vel = head_ed_vel

    head_ed_acc = np.diff(head_ed_vel, prepend=np.nan) 
    # ----------------------------- Calculate relative head movement -----------------------------

    head_rel_x = head_x - data['Neck_x']
    head_rel_y = head_y - data['Neck_y']

    headRel_ed_pos = np.sqrt(head_rel_x**2 + head_rel_y**2)
    headRel_ed_vel = np.sqrt(np.diff(head_rel_x, prepend=np.nan)**2 + 
                                np.diff(head_rel_y, prepend=np.nan)**2)

    #headRel_ed_vel = headRel_ed_vel / delta_t

    headRel_ed_acc = np.diff(headRel_ed_vel, prepend=np.nan) 

 
    # ----------------------------- Calculate body movement -----------------------------

    body_x = data[['Nose_x', 'Neck_x', 'RShoulder_x', 'LShoulder_x']].mean(axis=1)
    body_y = data[['Nose_y', 'Neck_y', 'RShoulder_y', 'LShoulder_y']].mean(axis=1)



    bodyUpper_ed_pos = np.sqrt(body_x**2 + body_y**2)
    bodyUpper_ed_vel = np.sqrt(np.diff(body_x, prepend=np.nan)**2 + 
                                  np.diff(body_y, prepend=np.nan)**2)
    
    bodyUpper_ed_acc = np.diff(bodyUpper_ed_vel, prepend=np.nan) 
    


    # ----------------------------- Sum of the Euclidean norms of velocity vectors -----------------------------
   # ----------------------------- Full Body Movement -----------------------------
    keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'REye', 'LEye']  # add more if available

    total_movement_ts = np.zeros(len(data))
    for kp in keypoints:
        x = data[f'{kp}_x'].values
        y = data[f'{kp}_y'].values
        
        # Frame-to-frame Euclidean distances
        distances = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        
        total_movement_ts += distances

  # ----------------------------- All Head keypoints  -----------------------------

    head_keypoints = ['Nose', 'REye', 'LEye']

    head_total_movement_ts = np.zeros(len(data))
    for kp in head_keypoints:
        x = data[f'{kp}_x'].values - data['Neck_x'].values
        y = data[f'{kp}_y'].values - data['Neck_y'].values
        
        head_distances = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        head_total_movement_ts += head_distances

  # ----------------------------- All Body keypoints  -----------------------------
    
    body_keypoints = ['Neck', 'RShoulder', 'LShoulder']
    
    body_total_movement_ts = np.zeros(len(data))

    for kp in body_keypoints:
        x = data[f'{kp}_x'].values - data['Nose_x'].values
        y = data[f'{kp}_y'].values - data['Nose_y'].values
        body_distances = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        body_total_movement_ts += body_distances
        

    
    # ----------------------------- Euclidean norm of the sum of velocity vectors -----------------------------

    # ----------------------------- Net Head Movement -----------------------------
    head_keypoints = ['Nose', 'REye', 'LEye']

    # Initialize arrays for summed velocity components
    sum_vx = np.zeros(len(data))
    sum_vy = np.zeros(len(data))

    for kp in head_keypoints:
        x = data[f'{kp}_x'].values - data['Neck_x'].values
        y = data[f'{kp}_y'].values - data['Neck_y'].values
        
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        
        sum_vx += vx
        sum_vy += vy

    # Net head movement (magnitude of summed velocity vectors)
    head_movement_norm = np.sqrt(sum_vx**2 + sum_vy**2)

    # ----------------------------- Net Body Movement -----------------------------
    body_keypoints = ['Neck', 'RShoulder', 'LShoulder', 'Nose']

    # Initialize arrays for summed velocity components
    sum_vx = np.zeros(len(data))
    sum_vy = np.zeros(len(data))

    for kp in body_keypoints:
        x = data[f'{kp}_x'].values
        y = data[f'{kp}_y'].values

        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])

        sum_vx += vx
        sum_vy += vy

    # Net body movement (magnitude of summed velocity vectors)
    body_movement_norm = np.sqrt(sum_vx**2 + sum_vy**2)

    # ----------------------------- Net Full Body keypoints  -----------------------------
    all_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'REye', 'LEye']

    sum_vx = np.zeros(len(data))
    sum_vy = np.zeros(len(data))    

    for kp in all_keypoints:
        x = data[f'{kp}_x'].values
        y = data[f'{kp}_y'].values
        
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        
        sum_vx += vx
        sum_vy += vy

    # Net full body movement (magnitude of summed velocity vectors)
    full_body_movement_norm = np.sqrt(sum_vx**2 + sum_vy**2)

    return {
        "head_x": head_x,
        "head_y": head_y,
        "head_ed_pos": head_ed_pos,
        "head_ed_vel": head_ed_vel,
        "head_ed_acc": head_ed_acc,

        "head_rel_x": head_rel_x,
        "head_rel_y": head_rel_y,
        "headRel_ed_pos": headRel_ed_pos,
        "headRel_ed_vel": headRel_ed_vel,
        "headRel_ed_acc": headRel_ed_acc,

        "body_x": body_x,
        "body_y": body_y,
        "body_ed_pos": bodyUpper_ed_pos,
        "body_ed_vel": bodyUpper_ed_vel,
        "body_ed_acc": bodyUpper_ed_acc,
        
        "whole_all_vel": total_movement_ts,
        "head_all_vel": head_total_movement_ts,
        "body_all_vel": body_total_movement_ts,

        "head_movement_norm": head_movement_norm,
        "body_movement_norm": body_movement_norm,
        "full_body_movement_norm": full_body_movement_norm

    }

def calculate_metrics_for_dataframes(processed_data, columns_to_keep, window_size=25*30, window_overlap=0, diagnostics_plots=False):
    """
    Applies metric calculation to all pre-processed DataFrames.

    Args:
        processed_data (dict): Dictionary of file_name → DataFrame.
        keyframes_x, keyframes_y, keyframes_top_left_eye, keyframes_bottom_left_eye,
        keyframes_top_right_eye, keyframes_bottom_right_eye (list): Column sets used in metric calculations.

    Returns:
        dict: Dictionary where each key is a file name and value is a DataFrame of computed metrics.
    """
    # Initialize a dictionary to hold metrics DataFrames
    metrics_data_frames = {}
    # Create a set of columns to read based on the keypoints and suffixes
    suffixes = ['_x', '_y']
    #columns = {f"{kp}{suffix}" for kp in columns_to_keep for suffix in suffixes}
    columns = [f"{kp}{suffix}" for kp in columns_to_keep for suffix in suffixes]

    # Concatenate all DataFrames into one, drop missing values
    all_dfs = [df[columns] for df in processed_data.values()]
    X_raw = pd.concat(all_dfs, ignore_index=True).dropna().astype(np.float32)

    # Build global symmetric template
    global_template = build_symmetric_template(X_raw, expected_cols=columns, mode="none")
    symmetrization_mode = 'none'
    ref_lengths = None if symmetrization_mode == "nose" else compute_reference_limb_lengths(global_template, columns)

    # Define windowing function
    def get_window_indices(data_length, window_size, overlap):
        step = int(window_size * (1 - overlap))
        return [(start, start + window_size) for start in range(0, data_length - window_size + 1, step)]
    
    # Process each DataFrame
    for file_name, data in tqdm(processed_data.items(), desc="Calculating metrics"):

        window_size = window_size
        window_overlap = window_overlap

        window_indices = []
        retained_windows = []

        win_indices = get_window_indices(len(data), window_size, window_overlap)
        for w_idx, (start, end) in enumerate(win_indices):
            window = data.iloc[start:end].reset_index(drop=True)
            if window.isnull().any().any():
                continue

            # Apply filtering to the window
            # window = filter_data_safe_preserve_nans(window, fs=25, cutoff=10, order=4, audit=False)

            window = window[columns]

            # Align keypoints in the window to the global template
            aligned_X, _ = align_keypoints(
                window, columns,
                reference="Torso" if symmetrization_mode != "nose" else "Nose",
                template=global_template,
                use_procrustes=True,
                allow_rotation=True,
                mode="window"
            )

            if diagnostics_plots:
                # alignment diagnostics plot
                plot_alignment_diagnostics(global_template, [window], columns, align_keypoints, n_samples=2, procrustes=True)
                # time series plot
                plot_timeseries_before_after(window, aligned_X, columns, file_name, w_idx)

            # Reshape poses for further processing
            poses = aligned_X.reshape(-1, len(columns)//2, 2)
            if ref_lengths:
                poses = batch_apply_fixed_lengths(poses, ref_lengths)

            # Collect the retained windows and their indices
            retained_windows.append(poses.reshape(-1, len(columns)))
            window_indices.append((w_idx, start, end))
        
        if not retained_windows:
            print(f"No valid windows found for: {file_name}")
            continue

        all_frames = np.vstack(retained_windows)
        mean_pose = all_frames.mean(axis=0)

        centered_windows = [w - mean_pose for w in retained_windows]
        
        all_metrics = []
        for (w_idx, start, end), w in zip(window_indices, centered_windows):
            window_df = rebuild_aligned_dataframe(w, columns)
            metrics = calculate_metrics(window_df, columns=columns)

            # Construct DataFrame
            metrics_data = pd.DataFrame(metrics)
            metrics_data["Window_Index"] = w_idx
            metrics_data["start"] = start
            metrics_data["end"] = end

            all_metrics.append(metrics_data)
    
        if all_metrics:
            parts = file_name.split("_")
            rename = f"{parts[1]}_{parts[0].upper()}_{'_'.join(parts[2:])}"
            metrics_data_frames[rename] = pd.concat(all_metrics, ignore_index=True)
        else:
            print(f"No Valid Windows found for: {file_name}")

    return metrics_data_frames


