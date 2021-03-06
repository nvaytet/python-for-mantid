from __future__ import division, print_function
import numpy as np
from scipy.ndimage import gaussian_filter1d

################################################################################
################################################################################
################################################################################

""" Wave-frame multiplication window edge finder

    Author: Neil Vaytet, European Spallation Source, <neil.vaytet@esss.se>
    Date: 11/2018


    This file contains two functions (see individual function code for a list of
    parameters):

    - detectPeaks(): A peak/valley finding routine provided by Marcos Duarte
                     (https://github.com/demotu/BMC). It was slightly modified
                     to disable the plotting functionalities.

    - getWFMWindows(): Starting from the positions of the valleys, the edges
                       of WFM windows are found using various thresholds.


    Examples:
    ---------

    1. Reading in a data file: the file must contain two columns: the first is
       TOF data (x) and the second is the amplitude (y). All other lines should
       be commented with a '#' sign so that `np.loadtxt()` ignores them.

       python get_wfm_windows.py --filename=spectrum.txt --plot

       This will print the window edges to standard output and produce an image
       figure.pdf showing the different variables that were used to find the
       window limits.


    2. Calling from another python script: you must have a 2D array containing
       the x and y data.

       left_edges, right_edges = getWFMWindows(data=my_2D_data_array, plot=True)


    3. Changing the thresholds: there are two different thresholds.

       - bg_threshold is used to find the global left and right edges of the
         signal over the background. To determine the background, we begin from
         the first (leftmost) point in the data and we iterate towards the
         right. This first point whose value exceeds
         `bg_threshold * (ymax - ymin)` (where `ymin` and `ymax` are the minimum
         and maximum y values in the entire data set) is selected as the leading
         edge. The trailing edge is found with the same procedure starting from
         the right end and we iterating towards the left.
         The default value is bg_threshold = 0.05.

       - win_threshold is used to find the left and right edges of each pulse
         frame. We iterate to one side starting from the valley center. We will
         first iterate towards the right, to find the leading edge of the next
         window. The mean y value between this valley and the next one (`mean`)
         is computed. The window edge is the first value that exceeds the a
         fraction of the mean: `y > win_threshold * mean`.
         The default value is win_threshold = 0.3.

       python get_wfm_windows.py --filename=spectrum.txt --bg_threshold=0.1 \
           --win_threshold=0.5

"""

################################################################################
################################################################################
################################################################################

# Peak detection routine by Marcos Duarte
#
# """Detect peaks in data based on their amplitude and other features."""
#
#
# __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
# __version__ = "1.0.5"
# __license__ = "MIT"
def detectPeaks(x, mph=None, mpd=1, threshold=0, edge="rising",
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def checkPeaks(peaks, nwindows):
    """
    This function check if the number of peaks found is the correct one.
    """
    nvalleys = len(peaks)-2
    if nvalleys != (nwindows - 1):
        print("Error: number of valleys should be {}! Found {}.".format(nwindows-1, nvalleys))
        return True # Continue looping
    else:
        print("OK: Found {} valleys.".format(nvalleys))
        return False # Exit the loop

################################################################################
################################################################################
################################################################################

def getWFMWindows(data=None, filename=None, nwindows=6, bg_threshold=0.05,
                    win_threshold=0.3, plot=False, gsmooth=0, xrange=None,
                    rebin_step_for_string_output=None):

    """
    Get the left and right edges of wave-frame multiplication windows.

    The parameters are:

    - filename: the text file to read containing the data

    - nwindows: the number of pulse windows (6 by default)

    - bg_threshold: the percentage above which y values are no longer considered
                    as background

    - win_threshold: the minimum percentage of window average to each when
                     searching for window edges

    - plot: plot figure if True

    - gsmooth: width of Gaussian kernel to smooth the data with. If gmsooth = 0
               (its default value), then a guess is performed based on the
               number of data points. If it is set to `None` then no smoothing
               is performed.

    - xrange: an array or tuple containing 2 elements which are the start and
              end TOF values to be considered for analysis. The data outside
              these bounds is ignored. You can set one of the limits to None
              if you want to have no limit on that end.
              For example: xrange=(1000,55000) or xrange=[5000,None]

    - rebin_step_for_string_output: a number. If specified, the function will
                                    return an array of strings that are to be
                                    used as the input for the Rebin algorithm.
                                    The number given as an argument will appear
                                    as the bin step in the middle of each range.
                                    For example, if the number supplied is 64,
                                    the output will look like:
                                    ['15000,64,20000', '25000,64,30000', ...]
                                    If unspecified, the function will return
                                    the indices (NOT the Tof values) of the left
                                    and right edges as arrays of values.
    """

    if data is None:
        if filename is not None:
            data = np.loadtxt(filename)
        else:
            raise RuntimeError("Either data or filename must be defined!")
    nx = np.shape(data)[0]
    print(nx)

    x = data[:,0]
    y = data[:,1]
    print(x)
    # Smooth the data with a gaussian filter
    if gsmooth == 0:
        gsmooth = max(int(nx/500),1)
    if gsmooth is not None:
        y = gaussian_filter1d(y, gsmooth)

    # If xrange is defined, then take that into account
    lbound = 0
    rbound = nx - 1
    if xrange is not None:
        if len(xrange) != 2:
            raise RuntimeError("xrange must contain exactly 2 elements")
        else:
            if xrange[0] is not None:
                for i in range(nx):
                    if x[i] >= xrange[0]:
                        lbound = i
                        break
            if xrange[1] is not None:
                for i in range(nx-1,1,-1):
                    if x[i] <= xrange[1]:
                        rbound = i
                        break

    # Find min and max values
    ymin = np.amin(y[lbound:rbound+1])
    ymax = np.amax(y[lbound:rbound+1])
    # Find the leading and trailing edges; i.e. the leftmost and rightmost
    # points that exceed the value `bg_threshold * (ymax - ymin)
    i_start = 0
    i_end = nx-1
    for i in range(lbound,nx):
        if y[i] > bg_threshold*(ymax - ymin):
            i_start = i
            break
    for i in range(rbound,1,-1):
        if y[i] > bg_threshold*(ymax - ymin):
            i_end = i
            break

    print(i_start,i_end, lbound, rbound)
    # Determine minimum peak distance (mpd):
    # We know there should be 6 windows between the leading and trailing edges.
    # Since the windows have approximately all the same size, we can estimate a
    # minimum peak distance to be close to the distance between leading and trailing
    # edges divided by 6 (but slightly less to be on the safe side).
    # Note that for the `detectPeaks` function, mpd is in units of data index, not
    # time-of-flight.
    mpd = int(0.75 * float(i_end - i_start) / nwindows)
    loop = True
    while loop:
        print("The minimum peak distance (mpd) is:",mpd)
        # Find valleys using `detectPeaks` function from Marcos Duarte
        peaks = detectPeaks(y, mpd=mpd, valley=True)
        # Now filter out peaks that are between start and end
        good_peaks = [i_start]
        for p in peaks:
            if (p > i_start+0.25*mpd) and (p < i_end-0.25*mpd):
                good_peaks.append(p)
        good_peaks.append(i_end)
        loop = checkPeaks(good_peaks, nwindows)

        # Try to remove peaks of there are too many
        if (len(good_peaks)-2) > (nwindows - 1):
            print("Too many peaks found. Attempting automatic dismissal")
            # if the value at the bottom of the valley is over the window
            # threshold, then this is not a real valley
            temp_peaks = good_peaks[:]
            for p in range(1,len(good_peaks)-1):
                rmean = np.average(y[good_peaks[p]:good_peaks[p+1]])
                lmean = np.average(y[good_peaks[p-1]:good_peaks[p]])
                mean = 0.5*(lmean + rmean)
                if y[good_peaks[p]] > (win_threshold*mean):
                    temp_peaks[p] = -100
                    print("Removed peak number {} at x,y = {},{}".format(p,x[temp_peaks[p]],y[temp_peaks[p]]))
            good_peaks = []
            for p in range(len(temp_peaks)):
                if temp_peaks[p] != -100:
                    good_peaks.append(temp_peaks[p])
            loop = checkPeaks(good_peaks, nwindows)

        # Try to reduce the minimum peak distance if there are too few peaks
        if (len(good_peaks)-2) < (nwindows - 1):
            print("Not enough peaks found. Reducing the minimal peak distance")
            mpd *= 0.75

    # Now for each valley, iterate to one side starting from the valley center and
    # find the window edge. We start from the first valley, which is the second
    # element of the `good_peaks` array because the first is the global leading
    # edge.
    # We first iterate towards the right, to find the leading edge of the next
    # window.
    # The mean y value between this valley and the next one (`mean`) is computed.
    # The window edge is the first value that exceeds the a fraction of the mean:
    # `y > win_threshold * mean`.

    # Define left and right window edges
    ledges = [i_start]
    redges = []

    for p in range(1,len(good_peaks)-1):

        # Towards the right ===================
        rmean = np.average(y[good_peaks[p]:good_peaks[p+1]])
        # Find left edge iterating towards the right
        for i in range(good_peaks[p]+1,good_peaks[p+1]):
            if (y[i] - y[good_peaks[p]]) >= (win_threshold*(rmean-y[good_peaks[p]])):
                ledges.append(i)
                break

        # Towards the left =======================
        lmean = np.average(y[good_peaks[p-1]:good_peaks[p]])
        # Find left edge iterating towards the right
        for i in range(good_peaks[p]-1,good_peaks[p-1],-1):
            if (y[i] - y[good_peaks[p]]) >= (win_threshold*(lmean-y[good_peaks[p]])):
                redges.append(i)
                break

    # Remember to append the global trailing edge
    redges.append(i_end)

    print("The frame boundaries are the following:")
    for i in range(len(ledges)):
        print("{}, {}".format(x[ledges[i]],x[redges[i]]))

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ["r","g","b","magenta","cyan","orange"]
        for i in range(len(ledges)):
            ax.add_patch(Rectangle((x[ledges[i]], ymin), (x[redges[i]]-x[ledges[i]]), (ymax-ymin), facecolor=colors[i%5], alpha=0.5))
        for p in range(len(good_peaks)-1):
            rmean = np.average(y[good_peaks[p]:good_peaks[p+1]])
            ax.plot([x[good_peaks[p]],x[good_peaks[p+1]]],[rmean,rmean],color="lime", lw=2)
            ax.plot([x[good_peaks[p]],x[good_peaks[p+1]]],[win_threshold*rmean,win_threshold*rmean], color="sienna", lw=2)
        bg = bg_threshold*(ymax - ymin)
        ax.plot(x, data[:,1], color="k", lw=2, label="Raw data")
        ax.plot(x, y, color="lightgrey", lw=1, label="Smoothed data")
        ax.plot([np.amin(x),np.amax(x)], [bg, bg], "--", color="pink", lw=1, label="Background threshold")
        ax.plot([x[good_peaks[0]],x[good_peaks[1]]], [-ymax,-ymax], color="lime", label="Window mean", lw=2)
        ax.plot([x[good_peaks[0]],x[good_peaks[1]]], [-ymax,-ymax], color="sienna", label="Window mean", lw=2)
        for p in range(1,len(good_peaks)-1):
            ax.plot(x[good_peaks[p]], y[good_peaks[p]], "o", color="r")
        ax.plot(x[good_peaks[0]], y[good_peaks[0]], "o", color="deepskyblue", label="Leading edge")
        ax.plot(x[good_peaks[-1]], y[good_peaks[-1]], "o", color="yellow", label="Trailing edge")
        ax.set_xlabel("Time-of-flight")
        ax.set_ylabel("Amplitude")
        ax.set_ylim([0.0,1.05*np.amax(data[:,1])])
        ax.plot(x[good_peaks[0]], -ymax, "o", color="r",label="Valleys")
        ax.legend(loc=(0,1.02),ncol=4, fontsize=7)
        fig.savefig("figure.pdf",bbox_inches="tight")

    # If we want output in Rebin format, then construct the array of strings
    if rebin_step_for_string_output is not None:
        output = []
        for i in range(len(ledges)):
            output.append("{},{},{}".format(x[ledges[i]],rebin_step_for_string_output,x[redges[i]]))
        return output
    # If not, the indices of the edges are returned, NOT the Tof values
    else:
        return [ledges, redges]

################################################################################
################################################################################
################################################################################

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Automatically detect wave-frame multiplication windows")

    parser.add_argument("-d", "--data", type=str, default=None, dest="data",
                        help="the 2D data array to process")

    parser.add_argument("-f", "--filename", type=str, default=None, dest="filename",
                        help="the name of the file to process")

    parser.add_argument("-n", "--nwindows", type=int, default=6, dest="nwindows",
                        help="the number of windows to be found")

    parser.add_argument("-b","--bg-threshold", type=float, default=0.05, dest="bg_threshold",
                        help="threshold above which we are no longer in"
                        "background signal, as a percentage of (ymax - ymin).")

    parser.add_argument("-w","--win-threshold", type=float, default=0.3, dest="win_threshold",
                        help="threshold to find window edge, as a percentage of"
                        "average signal inside window.")

    parser.add_argument("-g", "--gsmooth", type=int, default=0, dest="gsmooth",
                        help="the width of the Gaussian kernel to smooth the"
                        "data. If 0 (the default), then a guess is made based"
                        "on the resolution of the data set. If None, then no"
                        "smoothing is carried out.")

    parser.add_argument("-x", "--xrange", default=None, dest="xrange",
                        help="an array or tuple containing 2 elements which are"
                        "the start and end TOF values to be considered for"
                        "analysis. The data outside these bounds is ignored."
                        "You can set one of the limits to None if you want to"
                        "have no limit on that end. For example:"
                        "xrange=(1000,55000) or xrange=[5000,None].")

    parser.add_argument("-p", "--plot", action="store_true",
                        help="output the results to a plot")

    parser.add_argument("-r", "--rebin-step-for-string-output", type=str, default=None,
                        dest="rebin_step_for_string_output",
                        help="a number. If specified, the function will"
                             "return an array of strings that are to be"
                             "used as the input for the Rebin algorithm."
                             "The number given as an argument will appear"
                             "as the bin step in the middle of each range."
                             "For example, if the number supplied is 64,"
                             "the output will look like:"
                             "['15000,64,20000', '25000,64,30000', ...]"
                             "If unspecified, the function will return"
                             "the indices (NOT the Tof values) of the left"
                             "and right edges as arrays of values.")

    options = parser.parse_args()

    output = getWFMWindows(data=options.data,
                           filename=options.filename,
                           nwindows=options.nwindows,
                           bg_threshold=options.bg_threshold,
                           win_threshold=options.win_threshold,
                           gsmooth=options.gsmooth,
                           xrange=options.xrange,
                           rebin_step_for_string_output=options.rebin_step_for_string_output,
                           plot=options.plot)
