import os 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


def plot_rqa_line_pair(
    df,
    trial_col,
    pair_col,
    value_col,
    value_label="DV",
    trial_labels=None,
    pair_labels=None,
    figsize=(8,5)
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Compute mean, SEM, and count per trial × pair
    stats = (
        df.groupby([trial_col, pair_col])[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])

    # Define order
    trial_order = ['trial0', 'trial1', 'trial2']
    pair_order = ['actual', 'pseudo']
    stats[trial_col] = pd.Categorical(stats[trial_col], trial_order, ordered=True)
    stats[pair_col] = pd.Categorical(stats[pair_col], pair_order, ordered=True)
    stats = stats.sort_values([trial_col, pair_col])

    # X positions and labels
    x = np.arange(len(trial_order))
    x_labels = [trial_labels.get(t, t) for t in trial_order] if trial_labels else trial_order

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['skyblue', 'lightcoral']
    markers = ['o', 's']

    for j, pair in enumerate(pair_order):
        subset = stats[stats[pair_col] == pair]
        ax.errorbar(
            x,
            subset['mean'],
            yerr=subset['sem'],
            fmt='-o',
            capsize=4,
            label=pair_labels.get(pair, pair) if pair_labels else pair,
            color=colors[j % len(colors)],
            marker=markers[j % len(markers)],
            markersize=6,
            linewidth=2
        )

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Labels and style
    ax.set_ylabel(value_label)
    # ax.legend(title="Pair", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    

def batch_plot_rqa_params(
    dir, 
    obs, 
    metric, 
    embeddings=[2], 
    delays=[1], 
    radii=[0.1], 
    minline=5, 
    print_obs=False, 
    remove_couples=None, 
    count_err=False,
    collapse_windows=False,
    dv_label=None
):
    combos = [(e, d, r) for e in embeddings for d in delays for r in radii]
    n = len(combos)

    cols = min(n, 3)  # up to 3 columns
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)

    for ax, (embedding, delay, radius) in zip(axes.flat, combos):
        
        file = f'CrossRqa_win_delay{delay}_dim{embedding}_rad{radius}_minl{minline}.csv'

        df = pd.read_csv(os.path.join(dir, file))

        if remove_couples is not None:
            df = df[~df['couple'].isin(remove_couples)]

        columns = ['trial', f'{obs}_{metric}', 'window_index']
        data = df[columns]

        data = data[~(data['window_index'] == 13)]
        for condition in data['trial'].unique():
            cond_df = data[data['trial'] == condition]
            summary = (
                    data
                    .groupby('trial')[f'{obs}_{metric}']
                    .agg(['mean', 'sem', 'count'])
                    .reset_index()
                    ) 
            
            if collapse_windows:
                x_vals = np.arange(len(summary))
                ax.errorbar(
                    x_vals,
                    summary['mean'],
                    yerr=summary['sem'],
                    capsize=4,
                    marker='o',
                    linestyle='none',
                    color='black',
                    elinewidth=2.5
                )

                ax.set_xticks(x_vals)
                ax.set_xticklabels(summary['trial'])
                ax.set_xlabel('Trial')

                if dv_label is not None:
                    ax.set_ylabel(f'{dv_label}_{metric}')
                else:
                    ax.set_ylabel(f'{obs}_{metric}')
                ax.set_title(f'Emb={embedding} | Delay={delay} | Rad={radius}')
                ax.grid(True)
            
            else:

                x_vals = np.arange(len(summary['window_index']))
                ax.errorbar(
                    x_vals,
                    summary['mean'],
                    yerr=summary['sem'],
                    label=f'{condition}',
                    capsize=4,
                    marker='o'
                )

                ax.set_xlabel('Window Index')
                if dv_label is not None:
                    ax.set_ylabel(f'{dv_label}_{metric}')
                else:
                    ax.set_ylabel(f'{obs}_{metric}')
                ax.set_title(f'Emb={embedding} | Delay={delay} | Rad={radius}')


                ax.grid(True)
                ax.legend(title='Condition')
                ax.set_xticks(x_vals)

            if print_obs:
                print(f"\nCondition: {condition}")
                print(summary[['window_index', 'count']].rename(columns={'count': 'n_observations'}))

            if count_err:
                err_cols = df[[x for x in df.columns if x.endswith('err_code')]]
                for err in err_cols:
                    print(f"\nCondition: {condition}")
                    print(f"{err}: {df[err].sum()}")

    # remove unused subplots if any
    for ax in axes.flat[len(combos):]:
        ax.remove()

    plt.tight_layout()
    plt.show()