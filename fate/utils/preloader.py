"""
    Functions to help preload data.
"""
import math
from copy import deepcopy

def iter_ceil(total, size):
    return int(math.ceil(total*1.0/size))

def data_to_cacheline(data, words_per_line):
    """
        Create a list of lists.
    """

    return [deepcopy([deepcopy(data[base*words_per_line+i]) if(base*words_per_line+i < len(data)) else 0 for i in range(words_per_line)]) \
        for base in range(iter_ceil(len(data),words_per_line))]
