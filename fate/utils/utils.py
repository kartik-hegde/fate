import csv
import numpy as np

def write_csv(list_of_lists, path):
    """Write out CSV"""
    with open(path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for mylist in list_of_lists:
            wr.writerow(mylist)

def truncate_idx(time_series, time, is_tuple=False):
    """
        Index to truncate the series.
    """
    for idx,val in enumerate(time_series):
        if(val >= time):
            return idx
    return idx

def percentagify(lst):
    return [100*i for i in lst]

def average(lst):
    """Return average"""
    return sum(lst)/len(lst)

def smoothen(series, window=10):
    "Smoothen the Series"
    from scipy import signal
    win = signal.windows.hann(50)
    # win = np.array([1]*100)
    return signal.convolve(series, win, mode='same')/ sum(win)

def softmax(series):
    """Return softmax %"""
    return [round(100*(val/sum(series)),2) for val in series]

def reverse_cumsum(series):
    """Perform the reverse of cumulative sum"""
    return np.diff(series, prepend=0)