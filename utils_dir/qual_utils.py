import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from tqdm import tqdm
import matplotlib.cm as cm
from tabulate import tabulate
from utils_dir.preprocess_utils import *
import collections

def get_conf_df(pose_dir, cols, get_na = False, na_thresh = 0, store = False, meta_data=None):

    rows = []
    files = [f for f in os.listdir(pose_dir) if f.startswith("p1") and f.endswith(".csv")]
    for file in tqdm(files, desc="Processing files"):
        if file.startswith("p1") and file.endswith(".csv"):
            p1_pose = pd.read_csv(os.path.join(pose_dir, file))
            p1_length = len(p1_pose)
            p2_pose = pd.read_csv(os.path.join(pose_dir,file.replace('p1', 'p2')))
            p2_length = len(p2_pose)

            p1_pose = p1_pose.copy()
            p2_pose = p2_pose.copy()

            if p1_length == p2_length:

                p2_file = file.replace('p1', 'p2')
                p1 = file.split('_')[0]
                p2 = p2_file.split('_')[0]
                couple = file.split('_')[1]
                condition = file.split('_')[4]

                if meta_data is not None:
                    md = pd.read_csv(meta_data)
                    md = md.copy()
                    
                    matching_row = md[(md['couple'] == int(couple)) & (md['trial'] == condition)]

                    if not matching_row.empty:
                    
                        start_frame = int(matching_row.iloc[0]['start_frame'])

                        p1_pose = p1_pose.iloc[start_frame:].reset_index(drop=True)
                        p2_pose = p2_pose.iloc[start_frame:].reset_index(drop=True)



                else:
                    print(f"Cannot find matching meta data for {file}")

                p1_row = {
                    "couple": couple,
                    "person": p1,
                    "condition": condition,
                }
                p2_row = {
                    "couple": couple,
                    "person": p2,
                    "condition": condition,
                }
                
                p1_pose.fillna(0, inplace=True)
                p2_pose.fillna(0, inplace=True)

                for col in cols:
                    metric = col + "_confidence"

                    if get_na:
                        p1_count = 0
                        p2_count = 0

                        p1_max_na = 0
                        p2_max_na = 0

                        p1_sum_na = 0
                        p2_sum_na = 0

                        p1_min_na = 0
                        p2_min_na = 0

                        for value in p1_pose[metric]:
                            if value <= na_thresh:
                                p1_count += 1
                                p1_sum_na += 1
                                if p1_count > p1_max_na:
                                    p1_max_na = p1_count
                            else:
                                if p1_min_na==0:
                                    p1_min_na = p1_count
                                p1_count = 0
                        for value in p2_pose[metric]:    
                            if value <= na_thresh:
                                p2_count += 1
                                p2_sum_na += 1
                                if p2_count > p2_max_na:
                                    p2_max_na = p2_count
                            else: 
                                if p2_min_na==0:
                                    p2_min_na = p2_count
                                p2_count = 0 
                        
                        p1_row.update({
                            f"{metric}_max_na": p1_max_na,
                            f"{metric}_sum_na": p1_sum_na,
                            f"{metric}_min_na": p1_min_na,
                            f"{metric}_perc_na": (p1_sum_na/p1_length) * 100

                            })

                        p2_row.update({
                            f"{metric}_max_na": p2_max_na,
                            f"{metric}_sum_na": p2_sum_na,
                            f"{metric}_min_na": p2_min_na,
                            f"{metric}_perc_na": (p2_sum_na/p2_length) * 100
                            })

                    p1_row.update({
                        f"{metric}_mean": p1_pose[metric].mean(skipna=False),
                        f"{metric}_median": p1_pose[metric].median(skipna=False),
                        f"{metric}_std": p1_pose[metric].std(skipna=False),
                        f"{metric}_max": p1_pose[metric].max(skipna=False),
                        f"{metric}_min": p1_pose[metric].min(skipna=False),
                
                    })
                    # Compute statistics for p2
                    p2_row.update({
                        f"{metric}_mean": p2_pose[metric].mean(skipna=False),
                        f"{metric}_median": p2_pose[metric].median(skipna=False),
                        f"{metric}_std": p2_pose[metric].std(skipna=False),
                        f"{metric}_max": p2_pose[metric].max(skipna=False),
                        f"{metric}_min": p2_pose[metric].min(skipna=False),

                    })

                rows.append(p1_row)
                rows.append(p2_row)
            
            else:
                print(f"Unequal: {'_'.join(file.split('_')[2:4])}")

    return pd.DataFrame(rows)

def plot_body_kps (df, group = str('person'), measure='confidence_mean', show_val=False):

    # person = 'p1'
    measure = measure

    groups = df[group].unique()

    kp_labels = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist", 
        8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 
        12: "LHip", 13: "LKnee", 14: "LAnkle",
        15: "REye", 16: "LEye", 17: "REar", 18: "LEar",
        19: "LBigToe",
        20: "LSmallToe", 21: "LHeel",
        22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
    }

    # Keypoint coordinates (x, y)
    keypoints = {
        0: (0, 10), 1: (0, 9), 2: (-1, 9), 3: (-1.6, 7.7), 4: (-1.7, 6.3),
        5: (1, 9), 6: (1.6, 7.7), 7: (1.7, 6.3), 8: (0, 6.2), 9: (-1, 6.2),
        10: (-1, 5), 11: (-1, 3), 12: (1, 6.2), 13: (1, 5), 14: (1, 3),
        15: (-0.5, 10.5), 16: (0.5, 10.5), 17: (-1, 10), 18: (1, 10),
        19: (1.5, 2.5), 20: (1.7, 2.6), 21: (0.8, 2.8), 22: (-1.5, 2.5), 23: (-1.7, 2.6), 24: (-0.8, 2.8)
    }

    # Define limb connections
    limbs = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (11, 22), (22, 23), (11, 24),
        (8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21),
        (0, 15), (0, 16), (15, 17), (16, 18)
    ]

    for grps in groups:

        kp_cols = df.columns[df.columns.str.contains(measure)]

        mean_values = df[df[group] == grps][kp_cols].mean().values

       
        cmap = cm.get_cmap('plasma')


        norm = plt.Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 9))
        
        kps_val = {}

        for (start, end) in limbs:
            x = [keypoints[start][0], keypoints[end][0]]
            y = [keypoints[start][1], keypoints[end][1]]
            ax.plot(x, y, color='darkgrey', linewidth=3)

        for idx, (x, y) in keypoints.items():
            value = mean_values[idx]
            color = cmap(norm(value))
            if kp_labels[idx] == "RBigToe" or kp_labels[idx] == "LBigToe" or kp_labels[idx] == "LHeel":
                ax.text(x + 0.12, y - 0.14, kp_labels[idx], fontsize=8, color='black')
            elif kp_labels[idx] == "RSmallToe":
                ax.text(x - 0.15, y + 0.12, kp_labels[idx], fontsize=8, color='black')
            else: 
                ax.text(x + 0.1, y + 0.1, kp_labels[idx], fontsize=8, color='black')
            ax.plot(x, y, 'o', color=color, markersize=10)

            kps_val[kp_labels[idx]] = round(float(value), 3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label=f"{measure}")

        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{group}: {grps}")

        plt.show()

        if show_val:
            pairs = {}
            unpaired = []

            for kp, value in kps_val.items():
                if kp.startswith("L"):
                    keypoint = kp[1:]
                    pairs.setdefault(keypoint, {})["L"] = (kp, value)
                elif kp.startswith("R"):
                    keypoint = kp[1:]
                    pairs.setdefault(keypoint, {})["R"] = (kp, value)
                else:
                    unpaired.append((kp, value))

            rows = []
            for keypoint in sorted(pairs.keys()):
                l_kp, l_val = pairs[keypoint].get("L", ("", ""))
                r_kp, r_val = pairs[keypoint].get("R", ("", ""))
                rows.append([l_kp, l_val, r_kp, r_val])

            for kp, value in sorted(unpaired):
                rows.append([kp, value, "", ""])

            headers = ["Left Keypoint", "Value", "Right Keypoint", "Value"]
            print(tabulate(rows, headers=headers, tablefmt="grid"))

def plot_face_kps(df, group = str('person'), measure='confidence_mean'):

    coords = {'bottomCenterLip': (538.195, 417.154), 'centerNostril': (538.195, 396.913), 'chin': (537.274, 434.176), 
            'innerBottomCenterLip': (538.195, 409.794), 'innerTopCenterLip': (538.195, 409.794), 'leftBottomLip': (542.335, 416.694), 
            'leftChin': (548.775, 432.335), 'leftEdgeEyeLeft': (510.592, 364.25), 'leftEdgeEyeRight': (528.074, 365.63), 
            'leftHighCheek': (582.358, 377.131), 'leftInNostril': (533.594, 395.072), 'leftInner1EyeBrow': (556.596, 350.909), 
            'leftInnerBottomLip': (548.335, 409.794), 'leftInnerEyeBrow': (548.775, 353.209), 'leftInnerLip': (553.376, 409.334), 
            'leftInnerTopLip': (542.335, 409.334), 'leftLowerCheek': (577.758, 400.593), 'leftLowerEyeLeft': (516.113, 366.09), 
            'leftLowerEyeRight': (523.013, 366.09), 'leftLowerJaw': (557.976, 427.735), 'leftLowerLip': (549.695, 413.934), 
            'leftMiddleCheek': (579.138, 389.092), 'leftMiddleJaw': (567.637, 420.374), 'leftNostril': (528.994, 393.232), 
            'leftOuter1EyeBrow': (571.317, 351.829), 'leftOuterEyeBrow': (575.918, 357.809), 'leftOuterLip': (557.056, 407.953), 
            'leftPeakEyeBrow': (563.957, 350.449), 'leftPupil': (519.793, 362.87), 'leftTemple': (582.818, 365.17), 
            'leftTopEyeInner': (523.473, 361.49), 'leftTopEyeOuter': (516.113, 361.49), 'leftTopLip': (542.335, 405.193), 
            'leftUpperJaw': (574.078, 411.634), 'leftUpperLip': (549.695, 406.113), 'noseBelowBridge': (538.195, 379.891), 
            'noseBottom': (538.195, 387.712), 'noseBridge': (538.655, 371.61), 'noseTop': (538.655, 362.87), 
            'rightBottomLip': (532.214, 416.694), 'rightChin': (524.393, 431.875), 'rightEdgeEyeLeft': (549.235, 366.55), 
            'rightEdgeEyeRight': (567.177, 366.09), 'rightHighCheek': (491.731, 373.451), 'rightInNostril': (542.335, 395.532), 
            'rightInner1EyeBrow': (523.013, 347.688), 'rightInnerBottomLip': (532.674, 409.334), 'rightInnerEyeBrow': (530.834, 350.449), 
            'rightInnerLip': (523.013, 407.493), 'rightInnerTopLip': (532.674, 409.334), 'rightLowerCheek': (494.491, 398.293), 
            'rightLowerEyeLeft': (555.216, 368.39), 'rightLowerEyeRight': (561.656, 368.85), 'rightLowerJaw': (514.273, 427.275), 
            'rightLowerLip': (526.694, 413.474), 'rightMiddleCheek': (493.571, 386.332), 'rightMiddleJaw': (505.532, 419.914), 
            'rightNostril': (546.935, 395.072), 'rightOuter1EyeBrow': (505.992, 349.069), 'rightOuterEyeBrow': (501.392, 354.129), 
            'rightOuterLip': (519.793, 406.573), 'rightPeakEyeBrow': (514.273, 347.228), 'rightPupil': (557.976, 365.17), 
            'rightTemple': (491.271, 361.49), 'rightTopEyeInner': (555.676, 363.33), 'rightTopEyeOuter': (561.656, 363.79), 
            'rightTopLip': (533.134, 404.273), 'rightUpperJaw': (498.171, 409.334), 'rightUpperLip': (527.154, 405.653), 
            'topCenterLip': (538.195, 405.653)}
    
    face_kps = list(coords.keys())

    groups = df[group].unique()

    measure = 'confidence_mean'


    kp_cols = df.columns[
        df.columns.str.contains(measure) &
        df.columns.str.split('_').str[0].isin(face_kps)
    ]

    for grps in groups:

        mean_values = df[df[group] == grps][kp_cols].mean().values


        x_coord = []
        y_coord = []
        labels = []
        colors = []

        for i, kp in enumerate(face_kps):
            if kp in coords:
                x, y = coords[kp]
                x_coord.append(x)
                y_coord.append(y)
                labels.append(kp)
                colors.append(df[df[group] == grps][kp_cols].mean()[f'{kp}_{measure}']) 

        x = np.array(x_coord)
        y = np.array(y_coord)
        colors = np.array(colors)

        # Create a colormap
        cmap = cm.get_cmap('plasma')

        # Normalize mean values for color mapping
        norm = plt.Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))

        # Plot
        fig, ax = plt.subplots(figsize=(6, 8))

        scatter = ax.scatter(x, -y, c=colors, cmap=cmap, norm=norm, s=100, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label="Mean Confidence")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        ax.set_title(f"{group}: {grps}")
        plt.show()

def plot_missing_kps(df, kps, threshold=30, figsize=(12, 5)):

    def compute_counts(df, kps):
        result = {}
        for kp in kps:
                result[kp] = int((df[f"{kp}_confidence_max_na"] >= threshold).sum())
        return result


    groupings = {
        'Person': 'person',
        'Condition': 'condition',
        'Condition + Person': ['condition', 'person']
    }

    plt.figure(figsize=(14, 6))

    colors = ['purple', 'orange', 'green', 'blue', 'red', 'brown', 'cyan', 'magenta']
    linestyles = ['-', '--', '-.', ':']

    for i, (group_label, group_by) in enumerate(groupings.items()):
        plt.figure(figsize=figsize)
        
        # Group the DataFrame
        grouped = df.groupby(group_by)
        
        for j, (name, group_df) in enumerate(grouped):
            counts = compute_counts(group_df, kps)
            x = list(counts.keys())
            y = list(counts.values())

            label = str(name)
            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            plt.plot(x, y, marker='o', label=label, color=color, linestyle=linestyle)

        plt.xlabel('Keypoint')
        plt.ylabel('Count of Files (>30)')
        plt.title(f'{group_label}')
        plt.grid(True)
        plt.xticks(rotation=30, ha='right')
        plt.legend(title=group_label)
        plt.tight_layout()
        plt.show()

def get_missing_win(pose_dir, cols, conf_threshold=0, interpolate_max=25, window_size_sec=60, fps=25, overlap=True, meta_data=None):
    
    def sliding_window(df, window_size, step_size):
        """
        Generator that yields (start_idx, end_idx, df_window) over df rows.
        50% overlap is achieved if step_size equals window_size//2.
        """
        start = 0
        n = len(df)
        while start < n:
            end = start + window_size
            if end > n:
                end = n
            yield (start, end, df.iloc[start:end].copy())
            if end == n:
                break
            start += step_size

    results = []

    relevant_cols = []

    for c in cols:
        c_x = c+'_x'
        c_y = c+'_y'
        c_confidence = c+'_confidence'

        relevant_cols.append(c_x)
        relevant_cols.append(c_y)
        relevant_cols.append(c_confidence)


    # STEP 1) Get dataframe
    for file in tqdm(os.listdir(pose_dir), desc="Processing files"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(pose_dir, file), usecols=lambda col: col in relevant_cols)

            pose_df = df.copy()

            person = file.split('_')[0]
            couple = file.split('_')[1]
            condition = file.split('_')[4]
            
            file_info = {
                'person': person,
                'couple': couple,
                'condition': condition,
                }
            
            # STEP 2) Trim dataframe according to meta data 
            if meta_data is not None:
                md = pd.read_csv(meta_data)
                md = md.copy()
                
                matching_row = md[(md['couple'] == int(couple)) & (md['trial'] == condition)]

                if not matching_row.empty:
                    start_frame = int(matching_row.iloc[0]['start_frame'])

                    pose_df = pose_df.iloc[start_frame:].reset_index(drop=True)

            else:
                print(f"Cannot find matching meta data for {file}")

            
            # STEP 3) Clean and filter dataframe
            conf_cols = [col for col in pose_df.columns]

            # 3.1 Convert rows with corresponding confidence value < conf_thresh to NaN
            for conf_col in conf_cols:
                obs = conf_col.replace("_confidence", "")
                x_col = f"{obs}_x"
                y_col = f"{obs}_y"

                if x_col in pose_df.columns and y_col in pose_df.columns:
                    if conf_threshold is not None:
                        conf = pose_df[conf_col]
                        pose_df.loc[conf < conf_threshold, [x_col, y_col]] = np.nan
                    else:
                        pose_df.loc[(pose_df[x_col] == 0) & (pose_df[y_col] == 0), [x_col, y_col]] = np.nan
            
            
            pose_df = pose_df[[col for col in pose_df.columns if col.endswith('_x') or col.endswith('_y')]]

            # 3.2 Interpolate missing data using max sequence.
            pose_df = pose_df.interpolate(limit=interpolate_max, limit_direction='both')

            # STEP 4) Analyse data per window.
            # Window params
            window_size = int(window_size_sec * fps)
            if not overlap:
                step_size = window_size
            else:
                step_size = window_size // 2

            win_idx = 0
            for (start, end, df_win) in sliding_window(pose_df, window_size, step_size):
                win_idx+=1

                file_info = {
                'person': person,
                'couple': couple,
                'condition': condition,
                'window_index': win_idx,
                'start': start,
                'end': end,
                }

                for col in df_win.columns:
                    if col.endswith('_x'):
                        sum_na = df_win[col].isna().sum()
                        
                        file_info[col.split('_')[0]] = int(sum_na)

                #missing_per_col = df_win.isna().sum()

                #print(missing_per_col['Nose_x'])

                #print(missing_per_col)


                results.append(file_info) 
    
    dataframe = pd.DataFrame(results)

    return dataframe


def plt_missing_windows(dataframe, kps, figsize=(12, 5)):

    missing_windows = []
    missing_kps_total = []

    cols = ['couple','person', 'condition', 'window_index', 'start', 'end'] + kps

    pose_df = dataframe[cols]

    # print(len(pose_df))

    windows = pose_df['window_index'].unique()
    conditions = pose_df['condition'].unique()

    # print(conditions)

    for cond in conditions:
        num_windows = 0
        cond_df = pose_df[pose_df['condition']==str(cond)]
        
        store_win = []
        store_count = []
        missing_kps_total = []
        missing_kps = []
        missing_windows=[]

        for win in windows:
            window_df = cond_df[cond_df['window_index']==win]
            #window_df = window_df[window_df['couple']=='142']

            window_kps = window_df[kps]

            for index, row in window_df.iterrows():

                if not row.empty:
                    num_windows += 1
                    if row[kps].any() > 0:
                        missing_windows.append(int(win))
                        missing_kps = [kp for kp, val in row[kps].items() if val > 0]

                        missing_kps_total.append(missing_kps)

                        #for i in row[kps].get(row[kps] > 0):
                            #   print(win, missing_kps)

        missing_kps_total = np.concatenate(missing_kps_total)

        c = collections.Counter(missing_kps_total)

        counter = collections.Counter(missing_windows)

        counter = sorted(counter.items())

        # print(counter)

        # Create bar graph
        fig, ax = plt.subplots()
    
        for i in counter:
            window = str(i[0])
            count = int(i[1])

            store_win.append(window)
            store_count.append(count)

        # print(store_count)
        # print(store_win)

        print(f"Sum windows removed: {sum(store_count)} || Proportion Removed: %{round((sum(store_count)/num_windows)*100, 3)} || Total Windows: {num_windows}")

        ax.bar(store_win, store_count, color='pink')
        ax.set_title(f"Condition: {cond}")
        ax.set_ylabel('Total Number of windows removed')
        ax.set_xlabel('Window Index')
        plt.show()

def plt_kps_missing(dataframe, kps, figsize=(14, 6)):
    missing_windows = []
    missing_kps_total = []

    cols = ['couple','person', 'condition', 'window_index', 'start', 'end'] + kps

    pose_df = dataframe[cols]

    # print(len(pose_df))

    windows = pose_df['window_index'].unique()
    windows = windows.tolist()
    conditions = pose_df['condition'].unique()

    nested_dict = {}

    for i in windows:
        for kp in kps:
            nested_dict[f'{i}'] = {kp: []}


    for cond in conditions:
        num_windows = 0
        cond_df = pose_df[pose_df['condition']==str(cond)]
        
        missing_kps_total = []

        if cond == 'trial0':                  
            nested_dict = {window: {keypoint: 0 
                                    for keypoint in kps} 
                                    for window in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        else:
            nested_dict = {window: {keypoint: 0 
                                    for keypoint in kps} 
                                    for window in windows}
        


        for win in windows:
            window_df = cond_df[cond_df['window_index']==win]
            #window_df = window_df[window_df['couple']=='142']

            window_kps = window_df[kps]

            for index, row in window_df.iterrows():

                if not row.empty:
                    num_windows += 1
                    if row[kps].any() > 0:
                        missing_windows.append(int(win))
                        missing_kps = [kp for kp, val in row[kps].items() if val > 0]

                        missing_kps_total.append(missing_kps)

                        for kp in missing_kps:
                            nested_dict[win][kp] += 1
        
        print(f"Condition: {cond}")

        df = pd.DataFrame.from_dict(nested_dict, orient='index')
        df.index.name = 'window'
        df.reset_index(inplace=True)

        df = df.melt(id_vars='window', var_name='keypoint', value_name='count')

        plt.figure(figsize=figsize)

        for kp in df['keypoint'].unique():
            subset = df[df['keypoint'] == kp]
            plt.plot(subset['window'], subset['count'], label=kp)

        plt.xlabel('Window')
        plt.ylabel('Count of KeyPoints missing atleast once')
        plt.title(f'Condition: {cond}')
        plt.legend(title='Keypoint')
        plt.grid(True)
        plt.xticks(df['window'].unique())
        plt.show()


