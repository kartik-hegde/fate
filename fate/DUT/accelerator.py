"""
    This file describes the entire accelerator. (Top level file)
"""

import simpy
import sys
import os
import numpy as np

# Local Objects
from fate.DUT.cache import CacheModule
from fate.DUT.dram import DRAM
from fate.DUT.network import Network
from fate.DUT.processing_element import ProcessingElement
from fate.DUT.scheduler import Scheduler

class Accelerator:
    """ Top level module."""

    def __init__(self, env, parameters, logger):

        self.env = env
        self.num_PE = parameters.NUM_PE
        self.logger = logger
        self.parameters = parameters

        # Creates the program scheduler
        self.scheduler = Scheduler(env=env, parameters=parameters)
        print("Scheduler Instantiated.")
        # NoC PE <--> SideCar
        self.noc_pe_sidecar = Network(env,parameters.PE_SIDECAR_NOC_CACHELINES_PER_CYCLE,parameters.PE_SIDECAR_NOC_LATENCY, logger['PE_SIDECAR_NoC'])
        # NoC PE <--> PE
        self.noc_pe_pe = Network(env,parameters.PE_PE_NOC_CACHELINES_PER_CYCLE,parameters.PE_PE_NOC_LATENCY, logger['PE_PE_NoC'])
        # NoC SideCar <--> DRAM
        self.noc_sidecar_dram = Network(env,parameters.SIDECAR_DRAM_NOC_CACHELINES_PER_CYCLE,parameters.SIDECAR_DRAM_NOC_LATENCY, logger['SIDECAR_DRAM_NoC'])

        # Setup DRAM
        self.dram = DRAM(env=env, network_in=self.noc_sidecar_dram, size=parameters.DRAM_SIZE, access_granularity=parameters.ACCESS_GRANULARITY,
                    addr_width=parameters.ADDR_WIDTH, linesize=parameters.CACHELINE_SIZE, LATENCY=parameters.DRAM_ACCESS_LATENCY)

        # SideCar (shared across PEs)
        self.sidecar_cache = CacheModule(env=env, next_level_cache=self.dram, network_in=self.noc_pe_sidecar, network_out=self.noc_sidecar_dram, outstanding_queue_size=parameters.SIDECAR_OUTSTANDING_QSIZE,
                                HIT_LATENCY=parameters.SIDECAR_HIT_LATENCY, MISS_LATENCY=parameters.SIDECAR_MISS_LATENCY, ADDR_WIDTH=parameters.ADDR_WIDTH, ACCESS_GRANULARITY=parameters.ACCESS_GRANULARITY,
                                size=parameters.SIDECAR_SIZE, linesize=parameters.CACHELINE_SIZE, associativity=parameters.SIDECAR_ASSOCIATIVITY, logger=logger['SIDECAR_CACHE'], write_through=False,
                                point_of_coherency=True, name='SIDECAR')

        # Instantiate all the PEs
        self.PEs = [ProcessingElement(env=self.env, dram=self.dram, sidecar_cache=self.sidecar_cache, inter_pe_noc=self.noc_pe_pe, pe_sidecar_noc=self.noc_pe_sidecar, parameters=parameters, scheduler=self.scheduler, logger=logger['PE'][i], name='Worker'+str(i)) for i in range(parameters.NUM_PE)]
        print("PEs instantiated.")

        # Update workers
        workers = {}
        for idx,worker in enumerate(self.PEs):
            workers['Worker'+str(idx)] = worker
        for worker in self.PEs:
            worker.update_workers(workers)

    def preload_dram(self):
        """ Lets you load the DRAM."""
        raise NotImplementedError

    def preload_l3(self):
        """ Lets you preload the L3 cache."""
        raise NotImplementedError

    def check_completion(self):
        """Checks if everything is complete (Moves to accelerator later)"""
        while True:
            status = [worker.state for worker in self.workers]
            # print(status)
            # print(self.scheduler.check_completion())
            status = [(worker.state == 'DONE') for worker in self.workers]
            if(all(status)):
                yield self.env.timeout(1)
                break
            else:
                yield self.env.timeout(1)

        sys.exit("RUN COMPLETE")

    def run(self, pe_mask=None, run_id=0):
        """ 
        Run the simulations.
        
        Input:
            base_addrs : A list of length of number of PEs, indicating the base.
            pe_mask : Enable/Disable a PE.
            run_id: An integer to save the stats file name as.
        """
        processes = []

        # Based on the Mask supplied, enable PEs
        pe_mask= [1]*self.parameters.NUM_PE if(pe_mask is None) else pe_mask

        # Note the time of simulation start
        self.logger['start_sim'] = self.env.now

        # Create separate processes for each PE
        for idx,pe in enumerate(self.PEs):
            if(pe_mask[idx]):
                processes.append(self.env.process(pe.run()))

        # Run them all and save the logs.
        yield simpy.events.AllOf(self.env, processes)

        # End of simulation time is noted
        self.logger['end_sim'] = self.env.now

        # Simulation Complete. Save the results.
        print("\n\nRun Completed in {0}".format(self.logger['end_sim'] - self.logger['start_sim']))
        np.save(str(run_id) + "_stats.npy", self.logger)

if __name__ == "__main__":
    from fate.TB.tests.test_sanity import test_sanity

    test_sanity()