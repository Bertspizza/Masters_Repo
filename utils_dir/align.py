import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import math
import pandas as pd

# The following accesses trim_start_frame, extract_keypoints, norm_by_res, mask_low_confidence, interpolate_nans, and filter_data_safe_preserve_nans
from utils_dir.cleaning_utils import *
from utils_dir.keypoint_utils import *

# The rest of the required defs are defined below

def order_xy_pairs(columns):
    """
    Order columns so p1/p2 pairs are grouped consistently.
    Uses skeleton-like pairing logic instead of pure alphabetical.
    """
    ordered = []
    used = set()

    # Extract all base labels (without x/y)
    base_labels = sorted({c.rsplit("_", 2)[0] for c in columns if c.endswith(("_x", "_y"))})

    for base in base_labels:
        if base in used:
            continue

        if base.startswith("L") or base.startswith("R"):
            root = base[1:]  # strip L/R
            for side in ["L", "R"]:
                candidate = f"{side}{root}"
                x_col = f"{candidate}_x"
                y_col = f"{candidate}_y"
                if x_col in columns and y_col in columns:
                    ordered.extend([x_col, y_col])
                    used.add(candidate)
        else:
            # Midline keypoint
            x_col = f"{base}_x"
            y_col = f"{base}_y"
            if x_col in columns and y_col in columns:
                ordered.extend([x_col, y_col])
            used.add(base)

    return ordered

def build_symmetric_template(X_raw, expected_cols, mode="none"):
    """
    Build a global template with optional symmetrization.
    
    Parameters
    ----------
    X_raw : pd.DataFrame
        Concatenated raw keypoint data across participants/sessions.
    expected_cols : list
        List of column names (x/y offsets).
    mode : str
        Symmetrization mode:
        - "torso": symmetrize shoulders/hips only
        - "nose": center the nose horizontally
        - "full": symmetrize all p1/p2 pairs
        - "none": return raw mean template
    
    Returns
    -------
    np.ndarray
        Template as (n_points, 2) array.
    """
    n_points = len(expected_cols) // 2
    template = np.array([
        [X_raw[col_x].mean(), X_raw[col_y].mean()]
        for col_x, col_y in zip(expected_cols[::2], expected_cols[1::2])
    ])

    if mode == "none":
        return template

    mid_x = (template[:, 0].max() + template[:, 0].min()) / 2

    if mode == "nose":
        try:
            nose_idx = next(i for i, n in enumerate(expected_cols[::2]) if "Nose" in n)
        except StopIteration:
            raise ValueError("No Nose keypoint found in expected_cols for nose symmetrization")
        x_shift = mid_x - template[nose_idx, 0]
        template[:, 0] += x_shift
        return template

    if mode == "torso":
        torso_labels = ["Shoulder", "Hip"]
        for label in torso_labels:
            for side in ["L", "R"]:
                try:
                    i = next(idx for idx, n in enumerate(expected_cols[::2]) if f"{side}{label}" in n)
                    mirror_side = "R" if side == "L" else "L"
                    j = next(idx for idx, n in enumerate(expected_cols[::2]) if f"{mirror_side}{label}" in n)
                except StopIteration:
                    continue

                xi, yi = template[i]
                xj, yj = template[j]
                xi_ref = mid_x - (xi - mid_x)
                xj_ref = mid_x - (xj - mid_x)
                avg_x, avg_y = (xi_ref + xj) / 2, (yi + yj) / 2
                template[i] = [avg_x, avg_y]
                template[j] = [avg_x, avg_y]
        return template

    if mode == "full":
        for i, name in enumerate(expected_cols[::2]):
            if "L" in name:
                mirror_name = name.replace("L", "R")
            elif "R" in name:
                mirror_name = name.replace("R", "L")
            else:
                continue

            try:
                j = next(idx for idx, n in enumerate(expected_cols[::2]) if mirror_name in n)
            except StopIteration:
                continue

            xi, yi = template[i]
            xj, yj = template[j]
            xi_ref = mid_x - (xi - mid_x)
            xj_ref = mid_x - (xj - mid_x)
            avg_xi, avg_yi = (xi_ref + xj) / 2, (yi + yj) / 2
            avg_xj, avg_yj = (xj_ref + xi) / 2, (yj + yi) / 2
            template[i] = [avg_xi, avg_yi]
            template[j] = [avg_xj, avg_yj]
        return template

    raise ValueError(f"Unknown symmetrization mode: {mode}")

def plot_alignment_diagnostics(global_template, raw_windows, expected_cols, align_keypoints, 
                               n_samples=5, procrustes=True):
    """
    Overlay the global template with several raw and aligned trial mean skeletons.
    Chooses Nose or Torso reference based on available keypoints.
    """
    # Helper: pick reference automatically
    if any("Nose_x" in col for col in expected_cols):
        reference = "Nose" if set(["Nose_x", "Nose_y"]).issubset(expected_cols) and \
                    not any("Shoulder" in col or "Hip" in col for col in expected_cols) \
                  else "Torso"
    else:
        reference = "Torso"

    print(f"[INFO] Alignment diagnostics using reference: {reference}")

    n_points = len(expected_cols) // 2
    sample_indices = np.linspace(0, len(raw_windows)-1, min(n_samples, len(raw_windows))).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Before Alignment")
    axes[1].set_title("After Alignment")
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis("off")

    for idx in sample_indices:
        window = raw_windows[idx]
        trial_mean = window.mean().values.reshape(n_points, 2)
        axes[0].scatter(trial_mean[:, 0], trial_mean[:, 1], alpha=0.6)

        aligned_X, _ = align_keypoints(
            window, expected_cols,
            reference=reference,
            template=global_template,
            use_procrustes=procrustes
        )
        aligned_mean = aligned_X.mean(axis=0).reshape(n_points, 2)
        axes[1].scatter(aligned_mean[:, 0], aligned_mean[:, 1], alpha=0.6)

    for ax in axes:
        ax.scatter(global_template[:, 0], global_template[:, 1],
                   c="red", label="Global Template", s=30)

    axes[0].legend()
    plt.tight_layout()
    plt.show()

def compute_reference_limb_lengths(global_template, keypoint_names):
    """Compute fixed limb lengths from the global template pose."""
    ref_lengths = {}
    for side in ["L", "R"]:
        try:
            i_shoulder = next(i for i,n in enumerate(keypoint_names) if f"{side}Shoulder" in n and n.endswith("_x")) // 2
            i_elbow    = next(i for i,n in enumerate(keypoint_names) if f"{side}Elbow"    in n and n.endswith("_x")) // 2
            i_wrist    = next(i for i,n in enumerate(keypoint_names) if f"{side}Wrist"    in n and n.endswith("_x")) // 2
        except StopIteration:
            continue

        ref_lengths[(i_shoulder, i_elbow)] = np.linalg.norm(global_template[i_elbow] - global_template[i_shoulder])
        ref_lengths[(i_elbow, i_wrist)]   = np.linalg.norm(global_template[i_wrist] - global_template[i_elbow])
    return ref_lengths

def compute_procrustes_transform(template, trial_mean, allow_rotation=True):
    """Compute Procrustes transform, optionally without rotation."""
    template_c = template - template.mean(axis=0)
    trial_c = trial_mean - trial_mean.mean(axis=0)

    norm_template = np.linalg.norm(template_c)
    norm_trial = np.linalg.norm(trial_c)
    template_c /= norm_template
    trial_c /= norm_trial

    if allow_rotation:
        U, _, Vt = np.linalg.svd(trial_c.T @ template_c)
        R = U @ Vt
    else:
        R = np.eye(2)  # identity matrix, no rotation

    scale = norm_template / norm_trial
    t = template.mean(axis=0) - scale * trial_mean.mean(axis=0) @ R
    return R, scale, t

def align_keypoints(df, keypoint_names, reference="Torso",
                    template=None, use_procrustes=False,
                    allow_rotation=True, mode="trial",
                    flip_y=True):
    """
    Align keypoints for a trial or window.
    Includes optional y-flip to switch from OpenPose y-down to upp2 y-up.
    """
    n_points = len(keypoint_names) // 2
    coords_all = df.values.reshape(len(df), n_points, 2)

    if np.isnan(coords_all).any() or np.isinf(coords_all).any():
        raise ValueError(f"[SKIP] NaN/Inf detected in {mode} alignment.")

    # === Optional y-flip ===
    if flip_y:
        y_max = coords_all[..., 1].max()
        coords_all[..., 1] = y_max - coords_all[..., 1]

    # === Step 1: Reference subtraction (after flip) ===
    if reference == "Torso":
        torso_indices = [i//2 for i,n in enumerate(keypoint_names) 
                         if any(lbl in n for lbl in ["LShoulder","RShoulder","LHip","RHip"]) and n.endswith("_x")]
        ref_x = coords_all[:, torso_indices, 0].mean()
        ref_y = coords_all[:, torso_indices, 1].mean()
    else:  # Nose
        nose_idx = [i//2 for i,n in enumerate(keypoint_names) if "Nose" in n and n.endswith("_x")][0]
        ref_x = coords_all[:, nose_idx, 0].mean()
        ref_y = coords_all[:, nose_idx, 1].mean()

    coords_all[:, :, 0] -= ref_x
    coords_all[:, :, 1] -= ref_y

    if not use_procrustes:
        return coords_all.reshape(len(df), n_points*2), (n_points, 2)

    if template is None:
        raise ValueError("Template required when use_procrustes=True")

    # === Step 2: Procrustes alignment ===
    trial_mean = coords_all.mean(axis=0)
    try:
        R, scale, t = compute_procrustes_transform(template, trial_mean, allow_rotation)
        aligned_frames = np.array([scale * c @ R + t for c in coords_all])
    except Exception as e:
        raise ValueError(f"Skipping Procrustes due to error: {e}")

    return aligned_frames.reshape(len(df), n_points*2), (n_points, 2)

def batch_apply_fixed_lengths(poses, ref_lengths):
    """
    Apply fixed limb lengths (precomputed) to a pose.
    Vectorized limb normalization for all frames in a trial.
    poses: shape (n_frames, n_points, 2)
    """
    poses = poses.copy()
    for (i1, i2), target_len in ref_lengths.items():
        v = poses[:, i2] - poses[:, i1]                # shape (n_frames, 2)
        current_len = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.where(current_len > 1e-6, target_len / current_len, 1)
        poses[:, i2] = poses[:, i1] + v * scale
    return poses

def rebuild_aligned_dataframe(aligned_X, expected_cols):
    """
    Rebuild a DataFrame from aligned_X ensuring consistent keypoint ordering.
    aligned_X: np.ndarray of shape (n_frames, n_points*2)
    expected_cols: list of keypoint columns [kp1_x, kp1_y, kp2_x, kp2_y, ...]
    """
    n_points = len(expected_cols) // 2
    poses = aligned_X.reshape(-1, n_points, 2)

    data = {}
    for idx, base_label in enumerate(expected_cols[::2]):
        data[base_label] = poses[:, idx, 0]   # x
        data[expected_cols[2*idx+1]] = poses[:, idx, 1]  # y
    return pd.DataFrame(data)[expected_cols]

# Define skeleton connections for visualisation
SKELETON_CONNECTIONS = {
    "body": [
        # Torso
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("Neck", "MidHip"),
        ("MidHip", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("MidHip", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
    ],
    "arm": [
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
    ]
}

def get_skeleton_pairs(keypoint_names, sets=("body", "arm")):
    """ Get pairs of keypoint indices for skeleton connections.
    Returns a list of tuples (point_index1, point_index2) for each connection.  """
    pairs = []
    for set_name in sets:
        if set_name in SKELETON_CONNECTIONS:
            for kp1, kp2 in SKELETON_CONNECTIONS[set_name]:
                try:
                    i1 = next(i for i, n in enumerate(keypoint_names) if kp1 in n and n.endswith("_x"))
                    i2 = next(i for i, n in enumerate(keypoint_names) if kp2 in n and n.endswith("_x"))
                    pairs.append((i1 // 2, i2 // 2))  # convert column index → point index
                except StopIteration:
                    continue
    return pairs

def create_pm_animation(pca, X, keypoint_names, save_path, mean_vector,
                        n_components=3, n_frames=250, scale=2.0,
                        mode="sine",  # "sine" (synthetic oscillation) or "subject" (real scores)
                        velocity_mode=False, fps=60,
                        use_absolute=False, show_gray=True,
                        width_openpose=720, height_openpose=720,
                        width_video=1920, height_video=1080,
                        zoom_factor=1.0, ref_lengths=None,
                        flip_y=True, adjust_aspect=False):
    """
    Create PCA movement animations with auto-scaled axes, auto y-flip,
    and optional aspect-ratio correction for normalised coordinates.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    coords_per_point = 2
    n_points = len(keypoint_names) // coords_per_point

    # === Precompute scaling factors if using absolute coordinates ===
    if use_absolute:
        scale_x = width_video / width_openpose
        scale_y = height_video / height_openpose
        ref_x, ref_y = width_video / 2, height_video / 2
    else:
        scale_x = scale_y = ref_x = ref_y = None

    # === Build PC trajectories ===
    all_data = []
    pc_scores = pca.transform(X)
    for i in range(n_components):
        if mode == "subject":
            coeffs = pc_scores[:n_frames, i]
        elif mode == "sine":
            t = np.linspace(0, 2*np.pi, n_frames)
            coeffs = np.sin(t) * scale * np.std(pc_scores[:, i])
        else:
            raise ValueError("mode must be 'subject' or 'sine'")

        reconstruction = np.outer(coeffs, pca.components_[i])
        if velocity_mode:
            displacements = np.cumsum(reconstruction / fps, axis=0)
            positions = displacements + mean_vector
        else:
            positions = reconstruction + mean_vector

        anim_data = positions.reshape(n_frames, len(keypoint_names))

        # Reorder into (n_frames, n_points, 2)
        pairs = []
        for idx in range(0, len(keypoint_names), 2):
            pairs.append(anim_data[:, idx:idx+2])
        anim_data = np.stack(pairs, axis=1)

        if use_absolute:
            abs_anim = np.zeros_like(anim_data)
            abs_anim[:, :, 0] = anim_data[:, :, 0] * (width_openpose / 2) * scale_x + ref_x
            abs_anim[:, :, 1] = anim_data[:, :, 1] * (height_openpose / 2) * scale_y + ref_y
            anim_data = abs_anim

        all_data.append(anim_data)

    all_data = np.array(all_data)
    base_pose = mean_vector.reshape(n_points, coords_per_point)

    if use_absolute:
        abs_base = np.zeros_like(base_pose)
        abs_base[:, 0] = base_pose[:, 0] * (width_openpose / 2) * scale_x + ref_x
        abs_base[:, 1] = base_pose[:, 1] * (height_openpose / 2) * scale_y + ref_y
        base_pose = abs_base

    # === Auto y-flip ===
    if flip_y:
        def flip_y_axis(arr):
            ymax = arr[..., 1].max()
            if ymax > 10:   # pixel space
                return ymax - arr[..., 1]
            else:           # normalised [0,1]
                return 1 - arr[..., 1]
        all_data[..., 1] = flip_y_axis(all_data)
        base_pose[:, 1] = flip_y_axis(base_pose)

    # === Axis limits with zoom ===
    all_x = np.concatenate([all_data[:, :, :, 0].ravel(), base_pose[:, 0]])
    all_y = np.concatenate([all_data[:, :, :, 1].ravel(), base_pose[:, 1]])

    x_center = (all_x.max() + all_x.min()) / 2
    y_center = (all_y.max() + all_y.min()) / 2
    x_range = (all_x.max() - all_x.min()) * zoom_factor
    y_range = (all_y.max() - all_y.min()) * zoom_factor

    ax_min_x, ax_max_x = x_center - x_range / 2, x_center + x_range / 2
    ax_min_y, ax_max_y = y_center - y_range / 2, y_center + y_range / 2

    # === Plotting ===
    ncols = min(3, n_components)
    nrows = math.ceil(n_components / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    skeleton_pairs = get_skeleton_pairs(keypoint_names)
    animated_artists = []

    for i, ax in enumerate(axes[:n_components]):
        ax.set_xlim(ax_min_x, ax_max_x)
        ax.set_ylim(ax_min_y, ax_max_y)
        ax.set_aspect('equal')
        variance_explained = pca.explained_variance_ratio_[i] * 100
        ax.set_title(f"PM{i + 1}: {variance_explained:.1f}% var")
        ax.axis('off')

        if show_gray:
            gray_dots = ax.scatter(base_pose[:, 0], base_pose[:, 1],
                                   c='gray', s=8, alpha=0.5, zorder=1)
        else:
            gray_dots = None

        red_dots = ax.scatter([], [], c='black', s=8, zorder=2)
        red_lines = [ax.plot([], [], c='black', lw=1.5, zorder=1)[0] for _ in skeleton_pairs]
        gray_lines = [ax.plot([], [], c='gray', lw=1, alpha=0.5, zorder=0)[0] for _ in skeleton_pairs] if show_gray else []

        animated_artists.append((red_dots, gray_dots, red_lines, gray_lines))

    for ax in axes[n_components:]:
        ax.axis("off")

    def update(frame):
        artists = []
        for i in range(n_components):
            red_dots, gray_dots, red_lines, gray_lines = animated_artists[i]
            pose = all_data[i, frame]
            red_dots.set_offsets(pose)

            if show_gray and gray_dots is not None:
                gray_dots.set_offsets(base_pose)
                for (p1, p2), line in zip(skeleton_pairs, gray_lines):
                    line.set_data([base_pose[p1, 0], base_pose[p2, 0]],
                                  [base_pose[p1, 1], base_pose[p2, 1]])
                    artists.append(line)
                artists.append(gray_dots)

            for (p1, p2), line in zip(skeleton_pairs, red_lines):
                line.set_data([pose[p1, 0], pose[p2, 0]],
                              [pose[p1, 1], pose[p2, 1]])
                artists.append(line)
            artists.append(red_dots)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    ani.save(save_path, writer='ffmpeg', fps=40)
    plt.close()
    print(f"[DONE] Animation saved to {save_path}")

def compute_velocity(df, fps=25):
    """Compute velocity (dx/dt, dy/dt) for each keypoint column in df."""
    df_vel = df.diff().fillna(0) * fps
    df_vel.columns = [col.replace("_x", "_vx").replace("_y", "_vy") 
                      for col in df.columns]
    return df_vel

import matplotlib.pyplot as plt

def plot_timeseries_before_after(window_df, aligned_X, expected_cols, file_name, window_idx):
    """
    Plot time series before (grey) and after (blue) alignment.
    - window_df: original DataFrame (before alignment)
    - aligned_X: np.ndarray, shape (n_frames, n_points*2) after alignment
    - expected_cols: list of column names [kp1_x, kp1_y, ...]
    - file_name: identifier for saving/label
    - window_idx: index of the window
    """
    n_points = len(expected_cols) // 2
    aligned_df = pd.DataFrame(aligned_X, columns=expected_cols)

    fig, axes = plt.subplots(n_points, 2, figsize=(12, 2*n_points), sharex=True)
    axes = axes.reshape(n_points, 2)

    for i, base in enumerate(expected_cols[::2]):
        x_col, y_col = base, expected_cols[2*i+1]

        # Plot X
        axes[i, 0].plot(window_df[x_col], color="grey", alpha=0.7, label="Before")
        axes[i, 0].plot(aligned_df[x_col], color="blue", label="After")
        axes[i, 0].set_ylabel(base.replace("_x",""))
        if i == 0:
            axes[i, 0].legend(loc="upper right")
        axes[i, 0].set_title("X")

        # Plot Y
        axes[i, 1].plot(window_df[y_col], color="grey", alpha=0.7)
        axes[i, 1].plot(aligned_df[y_col], color="blue")
        axes[i, 1].set_title("Y")

    plt.suptitle(f"{file_name} - Window {window_idx}: Before vs After Alignment", y=1.02)
    plt.tight_layout()
    plt.show()
