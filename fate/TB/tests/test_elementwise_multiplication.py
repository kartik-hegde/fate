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

def get_tensors():
    """Get example tensors"""
    a = create_vector(128, 0.25)
    b = create_vector(128, 0.25)
    z = []
    return a,b,z

def get_aligned(addr, line_size):
    if(addr%line_size !=0):
        return addr + (line_size-addr%line_size)
    else:
        return addr

def get_graph(tensors):

    # Create a random Graph.
    testGraph = Graph()
    program  = [
        'Root',
        'GetHandles '+str(tensors[0]), 
        'HandlesToCoords '+str(tensors[0]), 
        'GetHandles '+str(tensors[1]),
        'HandlesToCoords '+str(tensors[1]),  
        'Intersect', 
        'HandlesToValues '+str(tensors[0]), 
        'HandlesToValues '+str(tensors[1]), 
        'Compute',
        'Populate '+str(tensors[2]), 
        ]

    # Insert the parent
    testGraph.graph[0].payload = program[0]
    # Insert all the work
    testGraph.add_edges([0],[2,5],program[1])
    testGraph.add_edges([1],[5],program[2])
    testGraph.add_edges([0],[4,5],program[3])
    testGraph.add_edges([3],[5],program[4])
    testGraph.add_edges([1,2,3,4],[6,7,9],program[5])
    testGraph.add_edges([5],[8],program[6])
    testGraph.add_edges([5],[8],program[7])
    testGraph.add_edges([6,7],[9],program[8])
    testGraph.add_edges([5,8],[],program[9])
    testGraph.printGraph()

    """
    
            0
      ______|______
     |            |
     1-----  ---- 3
     |    |  |    |
     2    |  |    4
     |____|  |____|   
        |_______|
            |
        ____5________
        |      |    |
        |      6    7
        |      |____|    
        |         |
        |         8
        |_________|
              |
              9
              
    Current Assumption: Single stream between 2 operators
    """            

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
    tensors = get_tensors()
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

    """
    def test_correctness():
        # Only PE0 ran the workload
        pe_instance = accelerator_instance.PEs[0]

        # Preload
        value = yield env.process(pe_instance.l1_dcache.read(BASE))

        if(value != 16):
            print("Expected {0} for PE {1} at {3}, but got {2}".format(16, 0, value, BASE))
            sys.exit("\n\n\t\tNESTING TEST FAILED.")

        return None
        
    # Traversal Code
    proc = env.process(test_correctness())

    # Run until completion
    env.run(proc)
    """

    print("\n\n\t\t SANITY TEST PASSED.")

    return True

if __name__ == "__main__":
    test_elementwise_multiply()