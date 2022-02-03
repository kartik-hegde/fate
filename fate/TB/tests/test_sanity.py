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
from fate.utils.graph import Graph
from fate.utils.preloader import data_to_cacheline
from fate.DUT.logger import Logger

def get_graph():

    # Create a random Graph.
    testGraph = Graph()
    program  = [
        'Init', 
        'GetHandles A', 
        'GetCoords A', 
        'GetHandles B', 
        'GetCoords B', 
        'Intersect', 
        'Compute']

    # Insert the parent
    testGraph.graph[0].payload = program[0]
    # Insert all the work
    testGraph.add_edges([0],[2],program[1])
    testGraph.add_edges([1],[5],program[2])
    testGraph.add_edges([0],[4],program[3])
    testGraph.add_edges([3],[5],program[4])
    testGraph.add_edges([2,4],[6],program[5])
    testGraph.add_edges([5],[],program[6])
    testGraph.printGraph()

    """
    
        0
      __|__
     |    |
     1    3
     |    |
     2    4
     |____|
        |
        5
        |
        6
    """

    return testGraph

def test_sanity():

    # Architectural Parameters
    arch_parameters = Parameters()

    # The simulation environment
    env = simpy.Environment()

    # Create a logger
    logger_instance = Logger(arch_parameters)
    logger= logger_instance.logger

    print("Instantiating Accelerator")
    # Instantiate the accelerator
    accelerator_instance = Accelerator(env, arch_parameters, logger)

    def preload_program(env):
        """
            Preload scheduler with program
        """
        print("Preloading Scheduler with the FATE Task graph")
        graph = get_graph()
        # Preload
        yield env.process(accelerator_instance.scheduler.update_graph(graph))
    # Run Preload
    env.run(env.process(preload_program(env,)))
    """
    print("Preloading DRAM")

    def preload_DRAM(env):
        words = data_to_cacheline(range(8), arch_parameters.WORDS_PER_LINE)
        yield env.process(accelerator_instance.dram.preload(words,addr_space.get_shared_base()))
    env.run(env.process(preload_DRAM(env)))
    """

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
    test_sanity()