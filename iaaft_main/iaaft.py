#! /usr/bin/env python3
"""
iaaft - Iterative amplitude adjusted Fourier transform surrogates

        This module implements the IAAFT method [1] to generate time series
        surrogates (i.e. randomized copies of the original time series) which
        ensures that each randomised copy preserves the power spectrum of the
        original time series.

[1] Venema, V., Ament, F. & Simmer, C. A stochastic iterative amplitude
    adjusted Fourier Transform algorithm with improved accuracy (2006), Nonlin.
    Proc. Geophys. 13, pp. 321--328
    https://doi.org/10.5194/npg-13-321-2006

"""
# Created: Tue Jun 22, 2021  09:44am
# Last modified: Tue Jun 22, 2021  12:39pm
#
# Copyright (C) 2021  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de> This
# program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import numpy as np
from tqdm import tqdm


def surrogates(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs



import matplotlib.pyplot as plt
import numpy as np

def plot_window_surrogates(original_window, surrogates_list, n_to_plot=2, column=None):
    """
    This is a modified version found in example.py created to plot original and surrogates on a window-by-window basis.

    Plot original time series, surrogate time series, and power spectra
    for a single window.

    Parameters
    ----------
    original_window : pd.DataFrame
        Original windowed time series (one or multiple columns)
    surrogates_list : list of pd.DataFrame
        List of surrogate time series (same shape as original_window)
    n_to_plot : int
        Number of surrogate series to plot
    """
    # For simplicity, use the first column if multiple columns
    if column is not None: 
        col = column
        x = original_window[col].values
        n = len(x)
    else:
        col = original_window.columns[0]
        x = original_window[col].values
        n = len(x)

    # Compute original power spectrum
    p = np.square(np.abs(np.fft.fft(x)))
    freq = np.fft.fftfreq(n)

    # Convert surrogates_list to array
    ns = len(surrogates_list)
    xs = np.array([s[col].values for s in surrogates_list])
    ps = np.zeros((ns, n))
    for i in range(ns):
        ps[i] = np.square(np.abs(np.fft.fft(xs[i])))

    # Select random surrogates to plot
    idx = np.random.randint(ns, size=n_to_plot)
    clrs = ["coral", "teal", "goldenrod"]

    # --- Plot setup ---
    AXLABFS, TIKLABFS = 14, 11
    fig = plt.figure(figsize=[12., 6.])

    fig.suptitle(f"{col}", fontsize=16)

    ax1 = fig.add_axes([0.10, 0.55, 0.375, 0.35])
    ax2 = fig.add_axes([0.10, 0.10, 0.375, 0.35])
    ax3 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

    # Original time series
    ax1.plot(x, "steelblue", label="original")

    # Surrogate time series
    for j, i in enumerate(idx):
        ax2.plot(xs[i], c=clrs[j], label=f"surrogate #{i}", alpha=0.45)

    # Power spectra
    k = np.argsort(freq)
    k = k[int(len(k)/2):]  # positive frequencies
    ax3.plot(freq[k], p[k], "o-", mec="steelblue", mfc="none",
             label="original", alpha=0.45)
    for j, i in enumerate(idx):
        ax3.plot(freq[k], ps[i, k], "x-", mec=clrs[j], mfc="none",
                 label=f"surrogate #{i}", alpha=0.45)
    ax3.set_yscale("log")

    # Beautify
    for ax in fig.axes:
        leg = ax.legend(loc="upper right")
        for txt in leg.get_texts():
            txt.set_size(TIKLABFS)
        ax.tick_params(labelsize=TIKLABFS)
    for ax in [ax1, ax2]:
        ax.set_ylabel("Signal", fontsize=AXLABFS)
        ax.set_xlabel("Time", fontsize=AXLABFS)
    ax3.set_xlabel("Frequency", fontsize=AXLABFS)
    ax3.set_ylabel("Power", fontsize=AXLABFS)

    plt.show()

