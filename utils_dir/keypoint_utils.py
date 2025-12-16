
import pandas as pd
import numpy as np

def extract_keypoints(file_path_or_data, sets=["hand", "face", "body"], prob_threshold=0.4, max_interp_frames=25):
    """
    Extract and clean keypoints from the dataset based on confidence threshold and interpolation.

    Parameters:
    - file_path_or_data (str or pd.DataFrame): Path to the CSV file or a DataFrame.
    - sets (list): List of sets to include in the output. Options: "hand", "face", "body", "center_face", "arm".
    - prob_threshold (float): Confidence threshold below which values are set to NaN.
    - max_interp_frames (int): Max consecutive NaNs to interpolate over (linear).

    Returns:
    - pd.DataFrame: DataFrame containing cleaned keypoint data.
    """

    # Load data
    if isinstance(file_path_or_data, str):
        data = pd.read_csv(file_path_or_data)
    elif isinstance(file_path_or_data, pd.DataFrame):
        data = file_path_or_data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    PREDEFINED_SETS = {
        'face': ["Eye", "Pupil", "Chin", "Jaw", "Cheek", "Nostril", "Lip", "Temple", "Nose"], #"Ear"
        'center_face': ["Eye", "Pupil", "Chin", "Nostril", "Lip", "Nose"],
        'hand': ["wristBase", "Tip"],
        'arm': ["Shoulder", "Elbow"], #"Wrist"
        'body': ["LEye", "REye", "Nose", "Shoulder", "Neck"] #"Hip"
    }

    labels = []
    for s in sets:
        labels.extend(PREDEFINED_SETS.get(s, [s]))

    # Keypoints to match exactly (e.g., from body model)
    exact_prefix_labels = {"Nose", "LEye", "REye"}

    # xyprob_cols = [
    #     col for col in data.columns
    #     if any(lbl in col for lbl in labels) and any(sfx in col for sfx in ["_x", "_y", "_confidence"])
    # ]

    xyprob_cols = []
    for col in data.columns:
        if any(col.endswith(sfx) for sfx in ["_x", "_y", "_confidence"]):
            for lbl in labels:
                if lbl in exact_prefix_labels:
                    if col.startswith(f"{lbl}_"):
                        xyprob_cols.append(col)
                        break
                elif lbl in col:
                    xyprob_cols.append(col)
                    break

    df_extracted = data[xyprob_cols].copy()

    # Apply confidence threshold
    for label in labels:
        conf_cols = [c for c in df_extracted.columns if label in c and c.endswith("_confidence")]
        x_cols = [c for c in df_extracted.columns if label in c and c.endswith("_x")]
        y_cols = [c for c in df_extracted.columns if label in c and c.endswith("_y")]

        for conf_col, x_col, y_col in zip(conf_cols, x_cols, y_cols):
            conf = df_extracted[conf_col]
            df_extracted.loc[conf < prob_threshold, [x_col, y_col]] = np.nan

    # Interpolate missing values
    numeric_cols = df_extracted.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_extracted[col] = df_extracted[col].interpolate(
            method='linear', limit=max_interp_frames, limit_direction='both'
        )

    # Drop confidence columns
    df_extracted = df_extracted.drop(columns=[c for c in df_extracted.columns if c.endswith("_confidence")])

    return df_extracted
