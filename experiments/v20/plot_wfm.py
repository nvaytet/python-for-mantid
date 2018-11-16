from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit






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


# from scipy import signal
# xs = np.arange(0, np.pi, 0.05)
# data = np.sin(xs)
# peakind = signal.find_peaks_cwt(data, np.arange(1,10))
# peakind, xs[peakind], data[peakind]


data = np.loadtxt('spectrum2.txt')
nx = np.shape(data)[0]
print(nx)

fig = plt.figure()
ax = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

mpd = 40

lines = [18454,27522, 28645,37670,38836,46479,
         47731,54510, 56324,62680,64485,68890]


x = data[:,0]*1000.0
y = data[:,1]
ymax = np.amax(y)


# peakind = detect_peaks(y, mph=500, mpd=500, valley=True)
peaks = detect_peaks(y, mpd=mpd, valley=True)

print("Number of peask found:",len(peaks))

mean = np.average(y)

print(np.average(y))

# Find leading and trailing edges
# threshold = np.average(y) / 2.0
threshold = 0.1
nmin = int(nx/50) # 100


print(nmin)
# Find leading background
background = y[0:nmin]
for i in range(nmin,nx):
    if y[i] > threshold*(ymax - np.average(background)):
        i_start = i
        break
    else:
        np.append(background,y[i])
# Find trailing background
background = y[-nmin-1:-1]
for i in range(nx-nmin,1,-1):
    if y[i] > threshold*(ymax - np.average(background)):
        i_end = i
        break
    else:
        np.append(background,y[i])

# for i in range(nx):
#     if y[i] > threshold:
#         i_start = i
#         break
# for i in range(nx-1,1,-1):
#     if y[i] > threshold:
#         i_end = i
#         break

# Now filter out peaks that are between start and end
good_peaks = [i_start]
for p in peaks:
    if (p > i_start+mpd) and (p < i_end-mpd):
        good_peaks.append(p)
good_peaks.append(i_end)



# Average peak distance
dist = 0.0
for p in range(len(good_peaks)-1):
    dist += x[good_peaks[p+1]] - x[good_peaks[p]]
dist /= float(len(good_peaks)-1)

print("average dist",dist)

dist /= 8.0

# ax.plot(x[nmin],y[nmin],'o')


y2 = gaussian_filter1d(y, 10)
# peakind2 = detect_peaks(y2, mpd=500, valley=True)







#
# y2 = gaussian_filter1d(y, 5)
#
# # ax.plot(x, y)
# # ax.plot(x, y2,color='k')
#
# y3 = -y2 + 800
#
#
# peakind = signal.find_peaks_cwt(y3, np.arange(180,200))
#
# print(np.arange(1000,5000,100))

# y = -y + np.amax(y)

ax.plot(x, y)
# ax.plot(x, y2)
ax.plot([x[i_start],x[i_start]],[0,ymax],color='r')
ax.plot([x[i_end],x[i_end]],[0,ymax],color='r')
# ax.plot(x, y2)

grady = np.abs(np.gradient(y2))
# ax2.semilogy(x, np.abs(y3))


# z = y[:]
width = 10

ledges = [i_start]
redges = []

for p in range(1,len(good_peaks)-1):
    # x1 = x[p-pwidth]
    # x2 = x[p+pwidth]
    # y1 = y[p-pwidth]
    # y2 = y[p+pwidth]
    # grad = (y2-y1)/(x2-x1)
    # for i in range(p-pwidth+1,p+pwidth):
    #     z[i] = y1 + grad * (x[i] - x1)
    # ax.plot(x[p], y[p], 'o', color='k')
    ax.plot(x[good_peaks[p]], y[good_peaks[p]], 'o', color='k')

    # for j in range()

    # w1 = int((good_peaks[p] - good_peaks[p-1]) / width)
    # w2 = int((good_peaks[p+1] - good_peaks[p]) / width)
    #
    x1 = int(0.5*(good_peaks[p]+good_peaks[p-1]))
    x2 = int(0.5*(good_peaks[p]+good_peaks[p+1]))
    # # print(x[x1:x2])
    # # print(y[x1:x2])

    # Towards the right ===================

    # Find start of average window
    for j in range(good_peaks[p],good_peaks[p+1]):
        if abs(x[j] - x[good_peaks[p]]) > dist:
            iav1 = j
            break
    # Find end of average window
    for j in range(good_peaks[p+1],good_peaks[p],-1):
        if abs(x[j] - x[good_peaks[p+1]]) > dist:
            iav2 = j
            break
    # Find start of iteration
    for j in range(good_peaks[p],good_peaks[p+1]):
        if abs(x[j] - x[good_peaks[p]]) > 0.5*dist:
            iter_start = j
            break


    mean2 = np.average(y2[iav1:iav2])
    print("MEAN2 is", mean2)
    for j in range(iter_start,x2):
        if (np.abs(y[j] - mean2)/mean2 < 0.05) and grady[j] < 0.5:
            # ax.plot([x[j],x[j]],[0,800],color='lime')
            ledges.append(j)
            break

    # Towards the left =======================

    # Find start of average window
    for j in range(good_peaks[p-1],good_peaks[p]):
        if abs(x[j] - x[good_peaks[p-1]]) > dist:
            iav1 = j
            break
    # Find end of average window
    for j in range(good_peaks[p],good_peaks[p-1],-1):
        if abs(x[j] - x[good_peaks[p]]) > dist:
            iav2 = j
            break
    # Find start of iteration
    for j in range(good_peaks[p],good_peaks[p-1],-1):
        if abs(x[j] - x[good_peaks[p]]) > 0.5*dist:
            iter_start = j
            break

    mean1 = np.average(y2[iav1:iav2])
    print("MEAN1 is", mean1)
    for j in range(iter_start,x1,-1):
        if (np.abs(y[j] - mean1)/mean1 < 0.05) and grady[j] < 0.5:
            # ax.plot([x[j],x[j]],[0,800],color='cyan')
            redges.append(j)
            break

redges.append(i_end)


colors = ['r','g','k','magenta','cyan','purple']

for i in range(len(ledges)):
    ax.plot([x[ledges[i]],x[ledges[i]]],[0,800],color=colors[i])
    ax.plot([x[redges[i]],x[redges[i]]],[0,800],color=colors[i])



    #     grad = (y2[j+w2] - y2[j]) / (x[j+w2] - x[j])
    #     if grad

    # yy = y[x1:x2] - np.amin(y[x1:x2])
    # popt, pcov = curve_fit(func, x[x1:x2], yy, p0 = [np.amax(yy),x[good_peaks[p]],0.5*(x[x1]+x[x2])])
    # print(popt)
    # ym = func(x[x1:x2], popt[0], popt[1], popt[2])
    # ax.plot(x[x1:x2], yy, c='grey')
    # ax.plot(x[x1:x2], ym, c='r')
    # break



# #popt returns the best fit values for parameters of the given model (func)
# print popt
#
# ym = func(x, popt[0], popt[1], popt[2])
# ax.plot(x, ym, c='r', label='Best fit')
# ax.legend()


# for p in peakind2:
#     ax.plot(x[p], y2[p], 'o', color='r')

# ax.plot(x, z, color='magenta')


w = signal.savgol_filter(y, 101, 2)
# ax.plot(x, w, color='lime')


# for i in lines:
#     ax.plot([i,i],[0,800],color='red')

fig.savefig('figure.pdf',bbox_inches='tight')
