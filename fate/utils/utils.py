import csv

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