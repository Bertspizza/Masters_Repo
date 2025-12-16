from utils_rqa import norm_utils, plot_utils, output_io_utils
from utils_rqa import rqa_utils_cpp
import os
import numpy as np

def crossRQA(data1, data2, params, file_info=None):

    result_row = file_info.copy() if file_info else {}
    metrics = {}

    """
    Perform Cross Recurrence Quantification Analysis (RQA).
    Parameters:
        data1 (np.ndarray): Time series data 1.
        data2 (np.ndarray): Time series data 2.
        params (dict): Dictionary of RQA parameters:
                       - norm, eDim, tLag, rescaleNorm, radius, tmin, minl,
                         doPlots, plotMode, phaseSpace, doStatsFile
    """
    for col in data1 :

        dta1 = data1[col].values
        dta2 = data2[col].values

        # Normalize data
        dataX1 = norm_utils.normalize_data(dta1, params['norm'])
        dataX2 = norm_utils.normalize_data(dta2, params['norm'])

        # Compute distance maxtrix for RQA
        ds = rqa_utils_cpp.rqa_dist(dataX1, dataX2, dim=params['eDim'], lag=params['tLag'])

        #print("Dist matrix stats:", 
         #       "shape", ds["d"].shape,
          #      "min", np.nanmin(ds["d"]), 
           #     "max", np.nanmax(ds["d"]), 
            #    "NaNs?", np.isnan(ds["d"]).any())

        # Perform CRQA calculations
        try:
            td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(ds["d"], rescale=params['rescaleNorm'], rad=params['radius'], diag_ignore=params['tw'], minl=params['minl'], rqa_mode="cross")
        except Exception as ex:
            err_code = 1
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        # Print stats
        if err_code == 0:
            if params['showMetrics']:
                print(f"%REC: {float(rs['perc_recur']):.3f} | %DET: {float(rs['perc_determ']):.3f} | MaxLine: {float(rs['maxl_found']):.2f}")
                print(f"Mean Line Length: {float(rs['mean_line_length']):.2f} | SD Line Length: {float(rs['std_line_length']):.2f} | Line Count: {float(rs['count_line']):.2f}")
                print(f"ENTR: {float(rs['entropy']):.3f} | LAM: {float(rs['laminarity']):.3f} | TT: {float(rs['trapping_time']):.3f}")
                print(f"Vmax: {float(rs['vmax']):.2f} | Divergence: {float(rs['divergence']):.3f}") 
                print(f"Trend_Lower: {float(rs['trend_lower_diag']):.3f} | Trend_Upper {float(rs['trend_upper_diag']):.3f}")
        else:
            print("Error in RQA computation. Check parameters and data.")

        # Plot results
        # plotMode: 'none', 'rp', 'rp_timeseries',
        plot_mode = params.get('plotMode', 'none')
        do_plots = params.get('doPlots', False)

        if do_plots and plot_mode != 'none' and ('rp' in plot_mode or 'timeseries' in plot_mode):
            save_path = None
            if params.get('saveFig', False):
                save_dir = os.path.join('images', 'rqa')
                os.makedirs(save_dir, exist_ok=True)
                fn = f"crossRQA_plot_{file_info.get('couple','NA')}_{file_info.get('window_index','NA')}.png" if file_info else "crossRQA_plot.png"
                save_path = os.path.join(save_dir, fn)

            plot_utils.plot_rqa_results(
                dataX=dataX1,
                dataY=dataX2,
                td=td,
                plot_mode=plot_mode,
                point_size=params.get('pointSize', 2),
                save_path=save_path
            )

        # Write stats
        if params['doStatsFile']:
            output_io_utils.write_rqa_stats("CrossRQA", params, rs, err_code, file_info=file_info)
        
        metrics[f"{col}_REC"] = rs['perc_recur'] if err_code == 0 else 0.0
        metrics[f"{col}_DET"] = rs['perc_determ'] if err_code == 0 else 0.0
        metrics[f"{col}_MAXLINE"] = rs['maxl_found'] if err_code == 0 else 0.0
        metrics[f"{col}_ENTROPY"] = rs['entropy'] if err_code == 0 else 0.0
        metrics[f"{col}_MEANLINE"] = rs['mean_line_length'] if err_code == 0 else 0.0
        metrics[f"{col}_LAM"] = rs['laminarity'] if err_code == 0 else 0.0
        metrics[f"{col}_DIV"] = rs['divergence'] if err_code == 0 else 0.0 
        metrics[f"{col}_MAXL_POSS"] = rs['maxl_poss'] if err_code == 0 else 0.0
        metrics[f"{col}_err_code"] = err_code
    
    result_row.update(metrics)
    return result_row
