import csv

def write_csv(list_of_lists, path):
    """Write out CSV"""
    with open(path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for mylist in list_of_lists:
            wr.writerow(mylist)