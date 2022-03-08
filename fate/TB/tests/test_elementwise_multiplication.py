######################## SYSTEM TEST ########################
"""
    This test is meant to test the nested strands, where the parent strand regs are visible to 
    child process, if the regs are followed by a '.'
"""
###########################################################

import simpy
import sys
from copy import deepcopy

# Local Objects
from fate.parameters import Parameters
from fate.DUT.accelerator import Accelerator
from fate.DUT.addrSpace import AddrSpace
from fate.utils.graph import Graph
from fate.utils.preloader import data_to_cacheline
from fate.DUT.logger import Logger
from fate.utils.example_generator import create_vector
from fate.utils.graph_functions import reference_intersection

def get_tensors(length, density=0.25):
    """Get example tensors"""
    a_vec, a = create_vector(length, density)
    b_vec, b = create_vector(length, density)
    z = []
    z_ref = reference_intersection(a_vec, b_vec)
    return a,b,z,z_ref

def get_aligned(addr, line_size):
    if(addr%line_size !=0):
        return addr + (line_size-addr%line_size)
    else:
        return addr

def get_graph(tensors):

    """
    
     0-----  ---- 2
     |    |  |    |
     1    |  |    3
     |____|  |____|   
        |_______|
            |
        ____4________
        |      |    |
        |      5    6
        |      |____|    
        |         |
        |         7
        |_________|
              |
              8
              
    """   

    # Create a random Graph.
    testGraph = Graph()
    program  = [
        'GetHandles '+str(tensors[0]), 
        'HandlesToCoords '+str(tensors[0]), 
        'GetHandles '+str(tensors[1]),
        'HandlesToCoords '+str(tensors[1]),  
        'Intersect', 
        'HandlesToValues '+str(tensors[0]), 
        'HandlesToValues '+str(tensors[1]), 
        'Compute mul',
        'Populate '+str(tensors[2]), 
        ]

    # Insert all the work

    # Node 0: GetHandles 1
    parents = []
    parent_names = []
    children = [1, 4]
    children_connections = {1:[(0,0),], 4:[(0,1),]} # Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[0])
    testGraph.update_root(0)

    # Node 1: HandlesToCoords 1
    parents = [0]
    parent_names = ['HandlesA']
    children = [4]
    children_connections = {4:[(0,0),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[1])

    # Node 2: GetHandles 1
    parents = []
    parent_names = []
    children = [3, 4]
    children_connections = {3:[(0,0),], 4:[(0,3),]} # Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[2])
    testGraph.update_root(2)

    # Node 3: HandlesToCoords 1
    parents = [2]
    parent_names = ['HandlesB']
    children = [4]
    children_connections = {4:[(0,2),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[3])

    # Node 4: Intersect
    parents = [0,1,2,3]
    parent_names = ['CoordsA', 'HandlesA', 'CoordsB', 'HandlesB',]
    children = [5,6,8]
    children_connections = {5:[(1,0),], 6:[(2,0),], 8:[(0,0),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[4])

    # Node 5: HandlesToValues 1
    parents = [4]
    parent_names = ['HandlesIntersectedA']
    children = [7]
    children_connections = {7:[(0,0),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[5])

    # Node 6: HandlesToValues 2
    parents = [4]
    parent_names = ['HandlesIntersectedB']
    children = [7]
    children_connections = {7:[(0,1),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[6])

    # Node 7: Compute
    parents = [5,6]
    parent_names = ['ValuesIntersectedA', 'ValuesIntersectedB']
    children = [8]
    children_connections = {8:[(0,1),]} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[7])

    # Node 8: Populate
    parents = [4,7]
    parent_names = ['CoordsZ', 'ValuesZ']
    children = []
    children_connections = {} #  Represents which input of consumer is connected to
    testGraph.add_edges(parents,parent_names,children,children_connections, program[8])

    testGraph.printGraph()

    return testGraph

def test_elementwise_multiply():

    # Architectural Parameters
    arch_parameters = Parameters()

    # The simulation environment
    env = simpy.Environment()

    # Address space
    addr_space = AddrSpace(arch_parameters)

    # Create a logger
    logger_instance = Logger(arch_parameters)
    logger= logger_instance.logger

    print("Instantiating Accelerator")
    # Instantiate the accelerator
    accelerator_instance = Accelerator(env, arch_parameters, logger)

    # ---------- PRELOAD DRAM --------------#

    print("Preloading DRAM")

    # Get the tensors
    tensors = get_tensors(length=1024*1024, density=0.01)
    reference_result = reference_intersection(tensors[0], tensors[1])

    # Create base addresses
    MAX_TENSOR = 1024*1024*8
    # Base of the shared memory
    A_PTR = addr_space.get_shared_base()
    B_PTR = get_aligned(A_PTR + max(len(tensors[0])<<2, MAX_TENSOR), arch_parameters.CACHELINE_SIZE)
    Z_PTR = get_aligned(B_PTR + max(len(tensors[1])<<2, MAX_TENSOR), arch_parameters.CACHELINE_SIZE)
       
    def preload_DRAM(env):
        words = data_to_cacheline(tensors[0], arch_parameters.WORDS_PER_LINE)
        yield env.process(accelerator_instance.dram.preload(words,A_PTR))
        words = data_to_cacheline(tensors[1], arch_parameters.WORDS_PER_LINE)
        yield env.process(accelerator_instance.dram.preload(words,B_PTR))
    env.run(env.process(preload_DRAM(env)))

    # ---------- PRELOAD PROGRAM --------------#

    def preload_program(env):
        """
            Preload scheduler with program
        """
        print("Preloading Scheduler with the FATE Task graph")
        graph = get_graph((A_PTR, B_PTR, Z_PTR))
        # Preload
        yield env.process(accelerator_instance.scheduler.update_graph(graph))
    # Run Preload
    env.run(env.process(preload_program(env)))

    print("Running Workloads ...")

    # All PEs need to execute this.
    pe_mask = [1]*(arch_parameters.NUM_PE)
    proc = env.process(accelerator_instance.run(pe_mask))

    # Run until completion
    env.run(proc)

    def test_correctness():
        # Only PE0 ran the workload
        pe_instance = accelerator_instance.PEs[0]

        # Preload
        nnz = yield env.process(pe_instance.l1_dcache.read(Z_PTR))
        reference_result = tensors[-1]
        reference_nnz = len(reference_result)

        if(nnz != reference_nnz):
            print("Expected {0} for PE {1} at {3}, but got {2}".format(reference_nnz, 0, nnz, Z_PTR))
            print(reference_result)
            sys.exit("\n\n\t\tNESTING TEST FAILED.")

        return None
        
    # Traversal Code
    proc = env.process(test_correctness())

    # Run until completion
    env.run(proc)

    print("\n\n\t\t SANITY TEST PASSED.")

    # Print statistics
    logger_instance.postprocess()

    return True

if __name__ == "__main__":
    test_elementwise_multiply()