"""
    Generate examples for the testing.
"""

"""
    This file is to generate random sparse matrices of given density.
"""
from random import randint
from scipy.sparse import random
import csv
import sys
import os
import argparse
import numpy as np
import scipy
import scipy.io

class Tensor:
    def __init__(self):
        self.shape = []
        self.nnz = None
        self.points = []
        self.data = []

def get_density(tensor):
    """ Return density of the tensor """
    shape = tensor.shape
    dense_vals = 1.0
    for i in shape:
        dense_vals *= i
    return dense_vals/tensor.nnz

def create_tensor(dims, density, format='coo'):
    """
        Create an output tensor of CSR format.
        Inputs:
            dims: Dimensionality of the tensor as a tuple (e.g., (4,4)).
            density: Density of the tensor.
            outpath: Path to write the output to.
        Output:
            Tensor object
    """
    if(len(dims)<=2):
        # Genrate the matrix
        tensor = random(*dims, density, format)
        tensor.data = np.ones(tensor.data.shape)
    else:
        sys.exit("Tensors not supported yet.")

    return tensor

def read_tensor(read_path):
    """
        Read a .mtx and store as CSR.
        Inputs:
            inpath: Path to read from
            outpath: Path to write the output to.
        Output:
            Tensor
    """
    print("Reading Matrix from ", read_path)
    tensor = scipy.io.mmread(read_path).tocsr()

    return tensor

def write_tensor(tensor, outpath, format='coo'):
    """
        Write out an output tensor of COO format.
        Inputs:
            tensor: Tensor object
            outpath: Path to write the output to.
        Output:
            Writes output file
    """
    if(format == 'coo'):
        # Write out
        with open(outpath,"w") as f:
            writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            # Header
            # Matrix
            if(len(tensor.shape)==2):
                tensor = tensor.tocoo()
                # Vector
                if(tensor.shape[1] == 1):
                    writer.writerow([tensor.shape[0]])
                    for idx in range(tensor.nnz):
                        writer.writerow([tensor.row[idx], tensor.data[idx]])
                else:
                    writer.writerow([*tensor.shape])
                    for idx in range(tensor.nnz):
                        writer.writerow([tensor.row[idx], tensor.col[idx], tensor.data[idx]])
            else:
                sys.exit("Tensors not supported yet.")
    elif(format == 'mtx'):
        scipy.io.mmwrite(outpath, tensor)

def create_vector(size, density):
    """Returns a list of given size and density"""
    vector_coords = random(1, size, density, 'csr')
    vector_coords.sort_indices()
    vector_coords = list(vector_coords.indices)
    vector_vals = [1.0 for _ in range(len(vector_coords))]
    vector = list(zip(vector_coords, vector_vals))
    return [element for item in vector for element in item]