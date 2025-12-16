import pandas as pd
import numpy as np 
import statsmodels
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, shapiro, t
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from itertools import combinations

"""
Functions designed for the hypothesis testing and the Linear Mixed Models (LMMs)

Functions included:
    - r2_marginal_conditional: Computes marginal and conditional R² for LMMs.
    - assumpsions: Test Normality of residuals and random effects, homoscedasticity for LMM models.
    - plot_simple_slopes_conv_score_c: Plot simple slops.
    - emmeans_with_slopes_rm_1: Compute estimated marginal means, simple slopes, and slope contrasts.
"""

def r2_marginal_conditional(model):
    """Compute Nakagawa & Schielzeth (2013) marginal/conditional R² for LMMs."""
    # Fixed effects variance
    fitted_fixed = np.dot(model.model.exog, model.fe_params)
    var_fixed = np.var(fitted_fixed, ddof=0)

    # Random effects variance (sum of diagonal for multiple random effects)
    if hasattr(model, "cov_re"):
        var_random = np.sum(np.diag(model.cov_re))
    else:
        var_random = 0.0

    # Residual variance
    var_resid = model.scale

    # Marginal (fixed) and conditional (fixed + random) R²
    R2_m = var_fixed / (var_fixed + var_random + var_resid)
    R2_c = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    return R2_m, R2_c


def assumptions(df, DVs=['headRel_ed_vel_REC'], formula="~ C(trial) * (window_scaled + avg_percomm + avg_sat_c)"):
    for dv in DVs:
        full_formula = f"{dv}{formula}"
        print(f"\n-------------------- TESTING DV: {dv} --------------------")
        
        try:
            # Fit mixed model
            mod = smf.mixedlm(full_formula, df, groups=df["couple"]).fit(reml=True)
            fitted = mod.fittedvalues
            residuals = mod.resid
            
            # --- 1. Residuals vs Fitted ---
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=fitted, y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Fitted values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs Fitted")
            plt.show()
            
            # --- 2. Histogram of residuals ---
            plt.figure(figsize=(6,4))
            sns.histplot(residuals, kde=True)
            plt.title("Histogram of Residuals")
            plt.show()
            
            # Shapiro-Wilk test for normality of residuals
            stat, p = stats.shapiro(residuals)
            print(f"Shapiro-Wilk test for residuals: stat={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                print("Residuals look Gaussian (fail to reject H0)")
            else:
                print("Residuals NOT Gaussian (reject H0)")
            
            # D'Agostino-Pearson test
            stat, p = stats.normaltest(residuals)
            print(f"D'Agostino-Pearson test for residuals: stat={stat:.3f}, p={p:.3f}")
            
            # --- 3. Q-Q plot of residuals ---
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title("Q-Q plot of residuals")
            plt.show()
            
            # --- 4. Q-Q plot and normality test for random intercepts ---
            re_values = [v.values[0] for v in mod.random_effects.values()]
            stats.probplot(re_values, dist="norm", plot=plt)
            plt.title("Q-Q plot of random intercepts")
            plt.show()
            
            stat, p = stats.shapiro(re_values)
            print(f"Shapiro-Wilk test for random intercepts: stat={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                print("Random intercepts look Gaussian")
            else:
                print("Random intercepts NOT Gaussian")
            
                
        except Exception as e:
            print(f"Error fitting model for {dv}: {e}")


def plot_simple_slopes_conv_score_c(slopes_df, dfresid=None):
    """
    Parameters:
        * slopes_df : pd.DataFrame
            Must contain columns: 'conv_type', 'role', 'slope', 'se'.
        * dfresid : int, optional
            Residual degrees of freedom for t-distribution. 
            If None, uses normal approximation (z=1.96).
    """
    plt.figure(figsize=(8,5))

    # Bar plot without automatic error bars
    sns.barplot(
        data=slopes_df,
        x='conv_type',
        y='slope',
        hue='role',
        ci=None,
        palette='Set2'
    )

    # Determine critical t-value for 95% CI
    if dfresid is not None:
        tval = t.ppf(0.975, dfresid)
    else:
        tval = 1.96  # approximate for large df

    # Add error bars manually as 95% CI
    for i, row in slopes_df.iterrows():
        x_pos = i % len(slopes_df['conv_type'].unique())
        hue_offset = -0.2 if row['role'] in ['AG', 'Agent'] else 0.2
        yerr = row['se'] * tval
        plt.errorbar(
            x=x_pos + hue_offset,
            y=row['slope'],
            yerr=yerr,
            fmt='none',
            color='black',
            capsize=5
        )

    plt.axhline(0, color='gray', linestyle='--')
    plt.ylabel("Slope of conv_score_c")
    plt.xlabel("Conversation Type")
    plt.title("Simple Slopes of conv_score_c by Role and Conversation Type (95% CI)")
    plt.xticks(rotation=15)
    plt.legend(title='Role')
    plt.tight_layout()
    plt.show()



def emmeans_with_slopes_rm(
    model, factors, covariates=None, p_correction='fdr_bh', 
    comparison_type=None
):
    """
    Compute estimated marginal means, simple slopes, and slope contrasts 
    (differences between slopes) for a Linear Mixed Effects model
    in a repeated-measures (within-subject) design, including optional 
    filtered pairwise contrasts with multiple comparison correction.

    Parameters
    ----------
    model : statsmodels MixedLMResults
        Fitted Linear Mixed Effects model.
    factors : list of str
        Names of categorical factors for which to compute marginal means.
    covariates : list of str, optional
        Names of continuous covariates to compute simple slopes for.
    p_correction : str or None
        Method for multiple comparison correction ('fdr_bh', 'bonferroni', 'holm', None).
    comparison_type : str or None
        Preset types of pairwise contrasts:
        - 'all': all possible pairs (default)
        - 'role_only': compare TG vs AG only (within same conv_type)
        - 'conv_type_only': compare conv_type levels only (within same role)
        - 'role_within_conv': TG vs AG for each conv_type (same as role_only)
        - None: no filtering, all pairs included
    """
    import pandas as pd
    import numpy as np
    from itertools import product, combinations
    from patsy import build_design_matrices
    from scipy.stats import t
    from statsmodels.stats.multitest import multipletests

    # --- Build grid of all factor combinations ---
    grid = pd.DataFrame(list(product(*[model.model.data.frame[f].unique() for f in factors])),
                        columns=factors)

    # Covariates set to zero (mean)
    if covariates:
        for cov in covariates:
            grid[cov] = 0

    # Define the matrix 
    design_info = model.model.data.design_info
    X = build_design_matrices([design_info], grid)[0]
    X_df = pd.DataFrame(X, columns=design_info.column_names)

    # marginal means and SEs
    betas = model.fe_params
    cov_params = model.cov_params()
    cov_aligned = cov_params.reindex(index=betas.index, columns=betas.index, fill_value=0.0)
    grid["emmeans"] = X_df.values @ betas
    se = np.sqrt(np.einsum("ij,jk,ik->i", X_df.values, cov_aligned.values, X_df.values))
    dfresid = getattr(model, "df_resid", None)
    tval = t.ppf(0.975, dfresid) if dfresid is not None else 1.96
    grid["se"] = se
    grid["lower"] = grid["emmeans"] - tval * se
    grid["upper"] = grid["emmeans"] + tval * se

    # Types of pairwise contrasts
    if comparison_type == 'role_only' or comparison_type == 'role_within_conv':
        comparison_filter = lambda r1, r2: r1['conv_type']==r2['conv_type'] and r1['role']!=r2['role']
    elif comparison_type == 'conv_type_only':
        comparison_filter = lambda r1, r2: r1['role']==r2['role'] and r1['conv_type']!=r2['conv_type']
    elif comparison_type == 'conv_pseudo':
        comparison_filter = lambda r1, r2: r1['pairing']!=r2['pairing'] and r1['conv_type']==r2['conv_type']
    elif comparison_type == 'pair_only' or comparison_type == 'pair_within_trial': 
        comparison_filter = lambda r1, r2: r1['trial']==r2['trial'] and r1['pair']!=r2['pair']
    elif comparison_type == 'trial_only' or comparison_type == 'trial_within_pair': 
        comparison_filter = lambda r1, r2: r1['trial']!=r2['trial'] and r1['pair']==r2['pair']
    elif comparison_type == 'all' or comparison_type is None:
        comparison_filter = lambda r1, r2: True
    else:
        raise ValueError(f"Unknown comparison_type: {comparison_type}")

    # Compute pairwise contrasts
    pairwise_list = []
    for (i1, row1), (i2, row2) in combinations(grid.iterrows(), 2):
        if not comparison_filter(row1, row2):
            continue

        diff_vec = X_df.iloc[i1] - X_df.iloc[i2]
        diff = diff_vec.values @ betas
        se_diff = np.sqrt(diff_vec.values @ cov_aligned.values @ diff_vec.values.T)
        t_stat = diff / se_diff if se_diff > 0 else 0.0
        if dfresid is not None and se_diff > 0:
            p_val = 2 * (1 - t.cdf(abs(t_stat), dfresid))
        elif se_diff > 0:
            from scipy.stats import norm
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
        else:
            p_val = 1.0
        pairwise_list.append({
            **{f"{f}_1": row1[f] for f in factors},
            **{f"{f}_2": row2[f] for f in factors},
            "difference": diff,
            "se": se_diff,
            "t": t_stat,
            "p": round(p_val, 3)
        })

    pairwise_df = pd.DataFrame(pairwise_list)
    if not pairwise_df.empty and p_correction is not None:
        reject, pvals_corrected, _, _ = multipletests(pairwise_df["p"], alpha=0.05, method=p_correction)
        pairwise_df["p_corrected"] = pvals_corrected
        pairwise_df["significant"] = reject
        resid_sd = np.sqrt(model.scale)
        pairwise_df["cohens_d"] = pairwise_df["difference"] / resid_sd
    else:
        pairwise_df["p_corrected"] = pairwise_df.get("p", [])
        pairwise_df["significant"] = pairwise_df.get("p", []) < 0.05

    # Compute simple slopes for covariates
    slopes_list = []
    if covariates:
        for cov in covariates:
            # Check if covariate interacts with any factors
            has_interactions = any(
                (f"{cov}:" in name) or (f":{cov}" in name) for name in betas.index
            )

            if not has_interactions:
                
                slope = betas[cov] if cov in betas.index else 0.0
                if cov in betas.index:
                    se_slope = np.sqrt(cov_aligned.loc[cov, cov])
                    t_slope = slope / se_slope if se_slope > 0 else 0.0
                    if dfresid is not None and se_slope > 0:
                        pval = 2 * (1 - t.cdf(abs(t_slope), dfresid))
                    elif se_slope > 0:
                        from scipy.stats import norm
                        pval = 2 * (1 - norm.cdf(abs(t_slope)))
                    else:
                        pval = 1.0
                else:
                    se_slope, t_slope, pval = 0.0, 0.0, 1.0

                slopes_list.append({
                    "covariate": cov,
                    "slope": slope,
                    "se": se_slope,
                    "t": t_slope,
                    "p": round(pval, 3),
                    **{f: None for f in factors}
                })

            else:
                
                for _, row in grid.iterrows():
                    coef_names = [cov] if cov in betas.index else []
                    for f in factors:
                        name1 = f"{cov}:C({f})[T.{row[f]}]"
                        name2 = f"C({f})[T.{row[f]}]:{cov}"
                        if name1 in betas.index:
                            coef_names.append(name1)
                        if name2 in betas.index:
                            coef_names.append(name2)

                    # Higher-order interactions
                    for name in betas.index:
                        if (name.startswith(f"{cov}:") or name.endswith(f":{cov}")):
                            if all(f"[T.{row[f]}]" in name for f in factors if f in name):
                                if name not in coef_names:
                                    coef_names.append(name)

                    slope = betas[coef_names].sum() if coef_names else 0.0
                    if coef_names:
                        cov_sub = cov_aligned.loc[coef_names, coef_names]
                        se_slope = np.sqrt(
                            np.ones(len(coef_names)) @ cov_sub.values @ np.ones(len(coef_names)).T
                        )
                        t_slope = slope / se_slope if se_slope > 0 else 0.0
                        if dfresid is not None and se_slope > 0:
                            pval = 2 * (1 - t.cdf(abs(t_slope), dfresid))
                        elif se_slope > 0:
                            from scipy.stats import norm
                            pval = 2 * (1 - norm.cdf(abs(t_slope)))
                        else:
                            pval = 1.0
                    else:
                        se_slope, t_slope, pval = 0.0, 0.0, 1.0

                    slopes_list.append({
                        **{f: row[f] for f in factors},
                        "covariate": cov,
                        "slope": slope,
                        "se": se_slope,
                        "t": t_slope,
                        "p": round(pval, 3)
                    })

        slopes_df = pd.DataFrame(slopes_list)
        if p_correction is not None and not slopes_df.empty:
            reject, pvals_corrected, _, _ = multipletests(slopes_df["p"], alpha=0.05, method=p_correction)
            slopes_df["p_corrected"] = pvals_corrected
            slopes_df["significant"] = reject
        else:
            slopes_df["p_corrected"] = slopes_df["p"]
            slopes_df["significant"] = slopes_df["p"] < 0.05
    else:
        slopes_df = pd.DataFrame()

    # Compute slope contrasts
    slope_contrasts = []
    if not slopes_df.empty and covariates:
        for cov in covariates:
            df_cov = slopes_df[slopes_df["covariate"] == cov]

            for (i1, row1), (i2, row2) in combinations(df_cov.iterrows(), 2):
                # Apply same comparison filter to slope contrasts
                if not comparison_filter(row1, row2):
                    continue

                slope_diff = row1["slope"] - row2["slope"]

                # Build contrast vector for slopes
                def get_coef_names(row):
                    coef_names = [cov] if cov in betas.index else []
                    for f in factors:
                        if row[f] is not None:
                            name1 = f"{cov}:C({f})[T.{row[f]}]"
                            name2 = f"C({f})[T.{row[f]}]:{cov}"
                            if name1 in betas.index:
                                coef_names.append(name1)
                            if name2 in betas.index:
                                coef_names.append(name2)
                    return coef_names

                coef_names1 = get_coef_names(row1)
                coef_names2 = get_coef_names(row2)

                contrast = pd.Series(0.0, index=betas.index)
                for name in coef_names1:
                    contrast[name] += 1.0
                for name in coef_names2:
                    contrast[name] -= 1.0

                se_diff = np.sqrt(contrast @ cov_aligned @ contrast.T)
                t_stat = slope_diff / se_diff if se_diff > 0 else 0.0
                if dfresid is not None and se_diff > 0:
                    p_val = 2 * (1 - t.cdf(abs(t_stat), dfresid))
                elif se_diff > 0:
                    from scipy.stats import norm
                    p_val = 2 * (1 - norm.cdf(abs(t_stat)))
                else:
                    p_val = 1.0

                slope_contrasts.append({
                    **{f"{f}_1": row1[f] for f in factors},
                    **{f"{f}_2": row2[f] for f in factors},
                    "covariate": cov,
                    "slope_diff": slope_diff,
                    "se": se_diff,
                    "t": t_stat,
                    "p": round(p_val, 3)
                })

        slope_contrasts_df = pd.DataFrame(slope_contrasts)
        if not slope_contrasts_df.empty and p_correction is not None:
            reject, pvals_corrected, _, _ = multipletests(slope_contrasts_df["p"], alpha=0.05, method=p_correction)
            slope_contrasts_df["p_corrected"] = pvals_corrected
            slope_contrasts_df["significant"] = reject
        else:
            slope_contrasts_df["p_corrected"] = slope_contrasts_df["p"]
            slope_contrasts_df["significant"] = slope_contrasts_df["p"] < 0.05
    else:
        slope_contrasts_df = pd.DataFrame()

    return grid, slopes_df, pairwise_df, slope_contrasts_df


    - Legend shows all categories
    """
    valence_map = {'PosDirect':'Positive', 'PosIndirect':'Positive',
                   'NDirect':'Negative', 'NIndirect':'Negative'}
    type_map = {'PosDirect':'Direct', 'PosIndirect':'Indirect',
                'NDirect':'Direct', 'NIndirect':'Indirect'}
    
    df = df[df['pair']==datatype]
    slopes_df = slopes_df[slopes_df['pair']==datatype]
    slopes_df['valence'] = slopes_df['conv_type'].map(valence_map)
    slopes_df['type'] = slopes_df['conv_type'].map(type_map)
    
    emmeans_df = emmeans_df[emmeans_df['pair']==datatype]
    emmeans_df['valence'] = emmeans_df['conv_type'].map(valence_map)
    emmeans_df['type'] = emmeans_df['conv_type'].map(type_map)
    
    color_map = {'Positive':'green', 'Negative':'red'}
    linestyle_map = {'Direct':'-', 'Indirect':'--'}
    
    # Build prediction grid
    cov_range = np.linspace(df[covariate].min(), df[covariate].max(), 100)
    plot_data = []
    for _, row in slopes_df.iterrows():
        emmean_base = emmeans_df.loc[emmeans_df['conv_type']==row['conv_type'], 'emmeans'].values[0]
        se_emmean = emmeans_df.loc[emmeans_df['conv_type']==row['conv_type'], 'se'].values[0]
        se_slope = row['se']
        for val in cov_range:
            pred = emmean_base + row['slope'] * val
            se_line = np.sqrt(se_emmean**2 + (val**2)*se_slope**2)
            lower = pred - 1.96*se_line
            upper = pred + 1.96*se_line
            plot_data.append({
                'conv_type': row['conv_type'],
                'valence': row['valence'],
                'type': row['type'],
                covariate: val,
                'pred': pred,
                'lower': lower,
                'upper': upper
            })
    plot_df = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Keep track of handles for a single legend
    legend_handles = {}
    
    for i, conv in enumerate(slopes_df['conv_type']):
        ax = axes[i]
        subset = plot_df[plot_df['conv_type']==conv]
        line = ax.plot(subset[covariate], subset['pred'], 
                color=color_map[subset['valence'].iloc[0]], 
                linestyle=linestyle_map[subset['type'].iloc[0]], 
                lw=2)[0]
        ax.fill_between(subset[covariate], subset['lower'], subset['upper'], 
                        color=color_map[subset['valence'].iloc[0]], alpha=0.2)
        
        # Collect legend handles (one per category)
        label = f"{subset['valence'].iloc[0]} {subset['type'].iloc[0]}"
        if label not in legend_handles:
            legend_handles[label] = line
        
        # optional points
        obs = df[df['conv_type']==conv]
        ax.scatter(obs[covariate], obs[dv], color='black', alpha=0.3, s=20)
        
        if display: 
            ax.set_title(conv)
            # ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("AG - TG Difference")
            ax.set_ylabel("Dependent variable")

    if display:
        # Single legend for all subplots
        fig.legend(legend_handles.values(), legend_handles.keys(), loc='upper center', ncol=2, title="Category")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
