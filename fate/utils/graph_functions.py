from scipy.sparse import coo_matrix, rand
from scipy import io
import numpy as np
import csv
from copy import deepcopy
import sys
import math

class CSFMatrix:
    def __init__(self):
        self.indptr0 = None
        self.indptr1 = None
        self.indices0 = None
        self.indices1 = None
        self.vals = None
        self.shape = None


def read_graph(read_path, csr_return=False):
    """
        Read a COO matrix and return its CSR.
    """
    if('tsv' in read_path):
        with open(read_path, 'r') as f:
            reader =csv.reader(f, delimiter='\t')
            data = [list(map(int, row)) for row in reader]
        data = np.array([np.array(d) for d in data])
        # Convert from COO to CSR
        col, row, d = data[:,0], data[:,1], data[:,2]
        csr = coo_matrix((d, (row, col))).tocsr()
    elif('mtx' in read_path):
        csr = io.mmread(read_path).tocsr()
    elif('npy' in read_path):
        csr = np.load(read_path, allow_pickle=True).item()
    print("The graph has a shape of {0}, nnz: {1}, and densiy {2}".format(csr.shape, csr.nnz, csr.nnz/(csr.shape[0]*csr.shape[1])))
    if(csr_return):
        return csr
    else:
        # Return row pointer and col pointer
        return csr.indptr, csr.indices

def csr2csf(csr):
    """
        Read a CSR and return a CSF.
    """
    csf = CSFMatrix()
    indptr0, indices0 = csr.indptr, csr.indices
    indices1 = []

    for idx, val in enumerate(indptr0[:-1]):
        if(indptr0[idx] == indptr0[idx+1]):
            np.delete(indptr0, idx)
        else:
            indices1.append(idx)
    indptr1 = np.array([0, len(indices1)])

    csf.indptr0 = indptr0
    csf.indices0 = indices0
    csf.vals = csr.data
    csf.indptr1 = indptr1
    csf.indices1 = np.array(indices1)
    csf.shape = csr.shape

    return csf

def mtx2npy(read_path):
    if('mtx' in read_path):
        print("Reading", read_path)
        csr = io.mmread(read_path).tocsr()
        np.save(read_path.split('.')[0]+'.npy', csr)
        print("Success", read_path)

def npy2mtx(read_path):
    if('npy' in read_path):
        print("Reading", read_path)
        csr = np.load(read_path, allow_pickle=True).item()
        io.mmwrite(read_path.split('.')[0]+'.mtx', csr)
        print("Success", read_path)

def symmetric2general(read_path):
    if('mtx' in read_path):
        print("Reading", read_path)
        csr = io.mmread(read_path).tocsr()
        io.mmwrite(read_path.split('.')[0]+'.mtx', csr, symmetry='general')
        print("Success", read_path)            

def tsv2mtx(read_path):
    """
        Read a COO matrix and return its CSR.
    """
    with open(read_path, 'r') as f:
        reader =csv.reader(f, delimiter='\t')
        data = [list(map(int, row)) for row in reader]
    data = np.array([np.array(d) for d in data])
    # Convert from COO to CSR
    col, row, d = data[:,0], data[:,1], data[:,2]
    coo = coo_matrix((d, (row, col)))
    write_path = read_path.split('.')[0]+'.mtx'
    io.mmwrite(write_path, coo)
    print("Wrote to ", write_path)

def create_vector(f, density=None):

    print("Working on ", f)
    csr_matrix =  read_graph(f, True)
    M,K = csr_matrix.shape
    density = csr_matrix.nnz/(csr_matrix.shape[0]*csr_matrix.shape[1]) if(density==None) else density
    csr_vector = rand(1, K, density, format='csr')
    name =  f[:-4] + '_V.npy'
    print("Saving to ", name)
    np.save(name, csr_vector)

def get_neighbors_sparse(row_ptr, col_id, node):
    base = row_ptr[node]
    bound = row_ptr[node+1]
    return list(col_id[base:bound])

def get_distance_sparse(row_ptr, dist, node):
    base = row_ptr[node]
    bound = row_ptr[node+1]
    return list(dist[base:bound])

def bfs_reference(adj_csr, start_node=0):
    """
        Reference function for BFS.
    """
    row_ptr, col_id = adj_csr[0], adj_csr[1]
    num_nodes = len(row_ptr)-1
    cur_frontier = get_neighbors_sparse(row_ptr, col_id, start_node)
    visited = [4294967295 for _ in range(num_nodes)]
    next_frontier = []
    next_frontier_filter = [0 for _ in range(num_nodes)]
    visited[start_node] = 0
    level = 1
    flag = True

    # Stats
    max_frontier_size = len(cur_frontier)
    neighbor_length = [len(cur_frontier),]

    while(flag):
        max_frontier_size = max(len(cur_frontier), max_frontier_size)
        # print("Current Frontier is, ", cur_frontier)
        for node in cur_frontier:
            # print(" \n\n ------ Node {0} --------- ".format(node))
            # print("Visited : {0} --------- ".format(visited[node]))            
            if(visited[node] == 4294967295):
                visited[node] = level
                # print("Current node not visited: {0}".format(node))
                neighbors = get_neighbors_sparse(row_ptr, col_id, node)
                neighbor_length.append(len(neighbors))
                for n in neighbors:
                    if((visited[n] == 4294967295) and (next_frontier_filter[n] == 0)):
                        next_frontier_filter[n] = 1
                        # print("Neighbor: {0} ".format(n))
                        next_frontier.append(n)
        # Set the flag
        flag = len(next_frontier)>0
        # Increment the level
        if(flag):
            level += 1
        # Swap the frontiers 
        cur_frontier = deepcopy(next_frontier)
        next_frontier_filter = [0 for _ in range(num_nodes)]
        next_frontier = []

    return (visited, level), (max_frontier_size, np.mean(neighbor_length), np.std(neighbor_length), sum(neighbor_length))

def connected_components(adj_csr, start_node=0):
    """
        This basically adds an outer loop over BFS.
        As many BFS unique runs you can do, that many components.
    """
    def bfs_cc(adj_csr, visited, start_node=0):
        """
            Reference function for BFS.
        """
        row_ptr, col_id = adj_csr[0], adj_csr[1]
        num_nodes = len(row_ptr)-1
        cur_frontier = get_neighbors_sparse(row_ptr, col_id, start_node)
        next_frontier = []
        next_frontier_filter = [0 for _ in range(num_nodes)]
        visited[start_node] = 0
        flag = True

        # Stats
        nodes =[]
        while(flag):
            # print("Current Frontier is, ", cur_frontier)
            for node in cur_frontier:
                # print(" \n\n ------ Node {0} --------- ".format(node))
                # print("Visited : {0} --------- ".format(visited[node]))            
                if(visited[node] == 4294967295):
                    nodes.append(node)
                    visited[node] = 0
                    # print("Current node not visited: {0}".format(node))
                    neighbors = get_neighbors_sparse(row_ptr, col_id, node)
                    for n in neighbors:
                        if((visited[n] == 4294967295) and (next_frontier_filter[n] == 0)):
                            next_frontier_filter[n] = 1
                            # print("Neighbor: {0} ".format(n))
                            next_frontier.append(n)
            # Set the flag
            flag = len(next_frontier)>0
            # Swap the frontiers 
            cur_frontier = deepcopy(next_frontier)
            next_frontier_filter = [0 for _ in range(num_nodes)]
            next_frontier = []

        return visited, nodes

    print("Starting CC run.")
    row_ptr, col_id = adj_csr[0], adj_csr[1]
    num_nodes = len(row_ptr)-1
    visited = [4294967295 for _ in range(num_nodes)]
    components = 0
    component_size = []

    for n in range(num_nodes):
        node = (start_node+n)%num_nodes
        # If the node is already visited, then no need to do BFS
        if(visited[node] == 4294967295):
            visited, nodes = bfs_cc(adj_csr, visited, start_node=node)
            components += 1
            component_size.append(nodes +[node,])
            # print("Completed walking through Component {0}".format(components))
    # print("Found {0} components, they were of size {1}".format(components, component_size))
    return components

def connected_components_scipy(path):
    from scipy.sparse.csgraph import connected_components
    n_components = connected_components(csgraph=read_graph(path, True), directed=False)
    print("SCIPY components, ", n_components)
    return n_components[0]

def sssp_reference(csr, start_node=1):
    """
        Reference function for BFS.
    """
    row_ptr, col_ptr, dist_ptr = csr.indptr, csr.indices, csr.data
    num_nodes = len(row_ptr)-1
    cur_frontier = [start_node,]
    next_frontier = []
    distance = [4294967295 for _ in range(num_nodes)]
    distance[start_node] = 0
    visited = [4294967295 for _ in range(num_nodes)]
    flag = True

    # Stats
    while(flag):
        # print("Current Frontier is, ", cur_frontier)
        for node in cur_frontier:
            # print("Visited : {0} --------- ".format(visited[node]))            
            if(visited[node] == 4294967295):
                # print(" \n\n ------ Node {0} --------- ".format(node))
                visited[node] =0
                cur_distance = distance[node]
                # print("Current node not visited: {0}".format(node))
                neighbors = get_neighbors_sparse(row_ptr, col_ptr, node)
                edges = get_distance_sparse(row_ptr, dist_ptr, node)
                for idx,n in enumerate(neighbors):
                    if(distance[n] > cur_distance + edges[idx]):
                        distance[n] = cur_distance + edges[idx]
                        # print("Updating distance of {0} to {1}".format(n, cur_distance + edges[idx]))
                    next_frontier.append(n)
        # Set the flag
        flag = len(next_frontier)>0
        # Swap the frontiers 
        cur_frontier = deepcopy(next_frontier)
        next_frontier = []

    return distance

def sssp_scipy(path, start_node=1):
    """
        SSSP Scipy
    """
    from scipy.sparse.csgraph import shortest_path
    dist_matrix = shortest_path(csgraph=read_graph(path, True), directed=False, indices=start_node, return_predecessors=False)
    return dist_matrix

def get_nnz_rows(row_ptr):
    result = []
    for idx, val in enumerate(row_ptr[:-1]):
        if(row_ptr[idx+1] > val):
            result.append(idx)
    return result

def spmspm_reference(csr_A, csr_B):
    """
        Reference function for BFS.
    """
    row_ptr_A, col_ptr_A, vals_A = csr_A.indptr, csr_A.indices, csr_A.data
    row_ptr_B, col_ptr_B, vals_B = csr_B.indptr, csr_B.indices, csr_B.data
    M,K,N = csr_A.shape[0], csr_B.shape[0], csr_B.shape[1]
    result_val = []
    result_row = []
    result_col = []

    flag = True

    # Stats
    num_muls = 0
    for a_row in get_nnz_rows(row_ptr_A):
        for b_col in get_nnz_rows(row_ptr_B):
            a_idx = 0
            b_idx = 0
            res = 0
            a_coords = get_neighbors_sparse(row_ptr_A, col_ptr_A, a_row)
            b_coords = get_neighbors_sparse(row_ptr_B, col_ptr_B, b_col)
            a_vals = get_distance_sparse(row_ptr_A, vals_A, a_row)
            b_vals = get_distance_sparse(row_ptr_B, vals_B, b_col)

            while((a_idx<len(a_coords)) and (b_idx<len(b_coords))):
                if(a_coords[a_idx] > b_coords[b_idx]):
                    b_idx += 1
                elif(a_coords[a_idx] < b_coords[b_idx]):
                    a_idx += 1
                elif(a_coords[a_idx] == b_coords[b_idx]):
                    res += a_vals[a_idx]*b_vals[b_idx]
                    a_idx += 1
                    b_idx += 1
                    num_muls += 1
                else:
                    sys.exit("Something wrong", a_coords[a_idx], b_coords[b_idx])
            if(res != 0):
                result_val.append(res)
                result_row.append(a_row)
                result_col.append(b_col)
    result = [result_row, result_col, result_val]
    result_coo = coo_matrix((result_val, (result_row, result_col)), shape=(M, N)).tocoo()
    return result, num_muls

def spmspv_reference(csf_A, csf_B):
    """
        Reference function for BFS.
    """
    row_ptr_A0, col_ptr_A0, row_ptr_A1, col_ptr_A1, vals_A = csf_A.indptr0, csf_A.indices0, csf_A.indptr1, csf_A.indices1, csf_A.vals
    row_ptr_B, col_ptr_B, vals_B = csf_B.indptr, csf_B.indices, csf_B.data
    M,K = csf_A.shape[0], csf_A.shape[1]
    result_val = []
    result_row = []
    result_col = []

    flag = True

    # Fix this
    b_coords = col_ptr_B

    # Stats
    num_muls = 0
    num_rows = row_ptr_A1[1]-row_ptr_A1[0]

    print(row_ptr_A1)
    print(row_ptr_A0[0],row_ptr_A0[1])
    print(row_ptr_B)
    # print("Total number of rows: {0}, nnz: {1}".format(len(col_ptr_A1), len(col_ptr_A0)))
    for a_row_idx, a_row in enumerate(col_ptr_A1):

            # if(len(col_ptr_A1)%(a_row_idx+1) == 100):
            #     print("{0}/{1} Complete.".format(a_row_idx, len(col_ptr_A1)))
            a_idx = 0
            b_idx = 0
            res = 0

            a_coords = get_neighbors_sparse(row_ptr_A0, col_ptr_A0, a_row_idx)

            while((a_idx<len(a_coords)) and (b_idx<len(b_coords))):
                if(a_coords[a_idx] > b_coords[b_idx]):
                    b_idx += 1
                elif(a_coords[a_idx] < b_coords[b_idx]):
                    a_idx += 1
                elif(a_coords[a_idx] == b_coords[b_idx]):
                    res += 1
                    a_idx += 1
                    b_idx += 1
                    num_muls += 1
            if(res != 0):
                result_val.append(res)
                result_row.append(a_row)
                result_col.append(0)
    result = [result_row, result_col, result_val]
    result_coo = coo_matrix((result_val, (result_row, result_col)), shape=(M, 1)).tocoo()
    return result_coo.tocsr(), num_muls

def spmspm_scipy(csr_A, csr_B):
    ref_result = csr_A.dot(csr_B.transpose())
    ref_result.sort_indices()
    print(ref_result.indptr, ref_result.indices, ref_result.data)

if __name__ == "__main__":
    if(sys.argv[1] == 'bfs'):
        (_, level), _ = bfs_reference(read_graph(sys.argv[2]), start_node=int(sys.argv[3]) if(len(sys.argv)>3) else 1) 
        print("Levels : {0}".format(level))
    elif(sys.argv[1] == 'cc'):
        components = connected_components(read_graph(sys.argv[2]), start_node=int(sys.argv[3]) if(len(sys.argv)>3) else 1) 
        from scipy.sparse.csgraph import connected_components
        n_components = connected_components(csgraph=read_graph(sys.argv[2], True), directed=False)
        print("SCIPY components, ", n_components)
        # print("Components : {0}".format(components))
    elif(sys.argv[1] == 'sssp'):
        # distances_scipy = sssp_scipy(sys.argv[2], start_node=int(sys.argv[3]) if(len(sys.argv)>3) else 1 )
        distances_ref = sssp_reference(read_graph(sys.argv[2], True), start_node=int(sys.argv[3]) if(len(sys.argv)>3) else 1)
        print(distances_ref)
    elif(sys.argv[1] == 'spmspm'):
        distances_ref = spmspm_reference(read_graph(sys.argv[2], True), read_graph(sys.argv[2], True),)
    elif(sys.argv[1] == 'spmspv'):
        csr_matrix =  read_graph(sys.argv[2], True)
        csf_matrix = csr2csf(csr_matrix)
        M,K = csf_matrix.shape
        # csr_vector = rand(1, K, density=float(sys.argv[3]), format='csr')
        csr_vector =  read_graph(sys.argv[3], True)
        result, num_muls = spmspv_reference(csf_matrix, csr_vector)
        # print(csr_matrix.toarray())
        # print(csr_vector.toarray())
        # print(result_coo.toarray())
        print("Total Multiplications: ", num_muls)

    elif(sys.argv[1] == 'mtx2npy'):
        mtx2npy(sys.argv[2])
    elif(sys.argv[1] == 'npy2mtx'):
        npy2mtx(sys.argv[2])
    elif(sys.argv[1] == 'tsv2mtx'):
        tsv2mtx(sys.argv[2])
    elif(sys.argv[1] == 'symmetric2general'):
        symmetric2general(sys.argv[2])
    elif(sys.argv[1] == 'matrix2vec'):
        create_vector(sys.argv[2],float(sys.argv[3]) if(len(sys.argv)>3) else None)
    
    else:
        print("Command Needed")