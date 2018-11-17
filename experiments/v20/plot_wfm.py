from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.ndimage import gaussian_filter1d
# from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle






# # %load ./../functions/detect_peaks.py
# """Detect peaks in data based on their amplitude and other features."""
#
# import numpy as np
#
# __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
# __version__ = "1.0.5"
# __license__ = "MIT"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

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
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

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

    x = np.atleast_1d(x).astype('float64')
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
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
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

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

# Gaussian function
def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def find_peaks(y):
    # Find valleys using `detect_peaks` function from Marcos Duarte
    peaks = detect_peaks(y, mpd=mpd, valley=True)
    # Now filter out peaks that are between start and end
    good_peaks = [i_start]
    for p in peaks:
        if (p > i_start+mpd) and (p < i_end-mpd):
            good_peaks.append(p)
    good_peaks.append(i_end)
    return good_peaks


################################################################################
################################################################################
################################################################################

# PARAMETERS ############
nwindows = 6
bg_threshold = 0.05
win_threshold = 0.05
plot = True
#########################

data = np.loadtxt('spectrum.txt')
nx = np.shape(data)[0]

fig = plt.figure()
ax = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)



x = data[:,0]*1000.0
y = gaussian_filter1d(data[:,1], 10)
ymin = np.amin(y)
ymax = np.amax(y)


# Find leading and trailing edges:
#
# Take the first `nmin` points starting from the left and compute a mean.
# This is the starting point for background.
# Then iterate towards the right. If the y value is higher than
# `threshold * (ymax - mean_background)` then this is the leading edge; if not
# the value is added to the background and we move to the next point.
#
# For the trailing edge, the same procedure is started from the right end and
# we iterate towards the left.
nmin = int(nx/50)
# Find leading background
background = y[0:nmin]
for i in range(nmin,nx):
    if y[i] > bg_threshold*(ymax - np.average(background)):
        i_start = i
        break
    else:
        np.append(background,y[i])
# Find trailing background
background = y[-nmin-1:-1]
for i in range(nx-nmin,1,-1):
    if y[i] > bg_threshold*(ymax - np.average(background)):
        i_end = i
        break
    else:
        np.append(background,y[i])

# Determine minimum peak distance (mpd):
# We know there should be 6 windows between the leading and trailing edges.
# Since the windows have approximately all the same size, we can estimate a
# minimum peak distance to be close to the distance between leading and trailing
# edges divided by 6 (but slightly less to be on the safe side).
# Note that for the `detect_peaks` function, mpd is in units of data index, not
# time-of-flight.
# y = gaussian_filter1d(y, 10)
mpd = int(0.75 * float(i_end - i_start) / nwindows)
print("The minimum peak distance (mpd) is:",mpd)
good_peaks = find_peaks(y)
print("Number of valleys found:",len(good_peaks)-2)
if (len(good_peaks)-2) != (nwindows - 1):
    print("Error: number of valleys should be %i!" % (nwindows-1))
    # print("Trying with a gaussian smoothing")
    # y = gaussian_filter1d(y, 10)
    # good_peaks = find_peaks(y)






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
    ax.plot(x[good_peaks[p]], y[good_peaks[p]], 'o', color='k')

    # Towards the right ===================
    rmean = np.average(y[good_peaks[p]:good_peaks[p+1]])
    ax.plot([x[good_peaks[p]],x[good_peaks[p+1]]],[rmean,rmean],color='lime')
    # Find left edge iterating towards the right
    for i in range(good_peaks[p],good_peaks[p+1]):
        if y[i] >= (win_threshold*rmean):
            ledges.append(i)
            break

    # Towards the left =======================
    lmean = np.average(y[good_peaks[p-1]:good_peaks[p]])
    ax.plot([x[good_peaks[p-1]],x[good_peaks[p]]],[lmean,lmean],color='lime')
    # Find left edge iterating towards the right
    for i in range(good_peaks[p],good_peaks[p-1],-1):
        if y[i] >= (win_threshold*lmean):
            redges.append(i)
            break


# Remember to append the global trailing edge
redges.append(i_end)

print("The frame boundaries are the following:")
for i in range(len(ledges)):
    print('{} --> {}'.format(x[ledges[i]],x[redges[i]]))


if plot:
    for i in range(len(ledges)):
        ax.add_patch(Rectangle((x[ledges[i]], ymin), (x[redges[i]]-x[ledges[i]]), (ymax-ymin), facecolor="C{}".format(i), alpha=0.5))
    for p in good_peaks:
        ax.plot(x[p], y[p], 'o', color='r')
    ax.plot(x, y, color="k")
    ax.set_xlabel("Time-of-flight")
    ax.set_ylabel("Amplitude")
    fig.savefig('figure.pdf',bbox_inches='tight')