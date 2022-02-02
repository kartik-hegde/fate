import simpy
import sys

from fate.DUT.buffet import Buffet
from fate.DUT.cache import CacheModule
from fate.DUT.network import Network

class ProcessingElement:

    def __init__(self, env, parameters, dram, sidecar_cache, inter_pe_noc, pe_sidecar_noc, scheduler, logger, name) -> None:
        """Instantiate"""
        self.env = env
        self.parameters = parameters
        self.dram = dram
        self.sidecar_cache = sidecar_cache
        self.inter_pe_noc = inter_pe_noc
        self.pe_sidecar_noc = pe_sidecar_noc
        self.logger = logger
        self.name = name
        self.scheduler = scheduler

        # Noc between Pipeline and L1 caches (Just wires, latency is included in the cache access itself)
        self.intra_pe_noc = Network(self.env,parameters.PE_L1_NOC_CACHELINES_PER_CYCLE, parameters.PE_L1_NOC_LATENCY, logger['PE_L1_NoC'])
        self.l1_l2_noc = Network(self.env,parameters.L1_L2_NOC_CACHELINES_PER_CYCLE, parameters.L1_L2_NOC_LATENCY, logger['L1_L2_NoC'])

        # l2 Cache
        self.l2_cache = CacheModule(env, next_level_cache=sidecar_cache, network_in=self.l1_l2_noc, network_out=self.inter_pe_noc,
                            outstanding_queue_size=parameters.L2_OUTSTANDING_QSIZE, HIT_LATENCY=parameters.L2_HIT_LATENCY,
                            MISS_LATENCY=parameters.L2_MISS_LATENCY, ADDR_WIDTH=parameters.ADDR_WIDTH, size=parameters.L2_SIZE,
                            ACCESS_GRANULARITY=parameters.ACCESS_GRANULARITY,linesize=parameters.CACHELINE_SIZE, associativity=parameters.L2_ASSOCIATIVITY,
                            write_through=True, logger=logger['L2_Cache'], name='L2-cache_PE'+str(name))

        # l1-d Cache
        self.l1_dcache = CacheModule(env, next_level_cache=self.l2_cache, network_in=self.intra_pe_noc, network_out=self.inter_pe_noc,
                            outstanding_queue_size=parameters.L1D_OUTSTANDING_QSIZE, HIT_LATENCY=parameters.L1D_HIT_LATENCY,
                            MISS_LATENCY=parameters.L1D_MISS_LATENCY, ADDR_WIDTH=parameters.ADDR_WIDTH, size=parameters.L1D_SIZE,
                            ACCESS_GRANULARITY=parameters.ACCESS_GRANULARITY, linesize=parameters.CACHELINE_SIZE, associativity=parameters.L1D_ASSOCIATIVITY,
                            write_through=True, logger=logger['L1_dCache'], name='L1-dcache_PE'+str(name))

        self.workers = None
        self.current_node_id = None
        self.current_consumer = None
        self.streaming_memory = Buffet(env, parameters, parameters.STR_MEMORY_SIZE)
        # Stages of any node: POLL -> STAGE -> EXECUTE -> COMPLETE
        self.state = 'POLL'

    def update_workers(self, workers):
        """Update Workers"""
        self.workers = workers

    def run(self):
        """
            Main control unit that runs the PE by transitioning through different stages.
        """
        while True:


            if(self.state == 'POLL'):
                # Ask for a node from the instruction queue (shared among PEs, one per cluster)
                if(self.scheduler.check_completion()):
                    yield self.env.timeout(1)
                    self.state = 'DONE'
                
                # Wait until a ready node is available
                yield (self.scheduler.ready_nodes_container.get(1) | self.env.timeout(10))
                self.current_node = yield self.env.process(self.scheduler.get_node(self.name))
                if(self.current_node == None):
                    print("Program Completion noted in PE{0}. Transitioning to DONE.".format(self.name))
                    self.state = 'DONE'
                else:
                    print("Got Node {0} at PE{1}. Transitioning to STAGE.".format(self.current_node.payload, self.name))
                    self.state = 'STAGE'
                yield self.env.timeout(1)

            # Stage the instruction, i.e., remains in staged until it is ready to execute and consumer is identified and ready.
            elif(self.state == 'STAGE'):
                yield self.env.timeout(1)
                consumer_ids = yield self.env.process(self.scheduler.get_consumer(self.current_node.name))
                # Terminal Node
                if(consumer_ids == None):
                    self.current_consumer = None
                    print("Terminal Node for node {0} at PE{1}. Transitioning to EXECUTE.".format(self.current_node.payload, self.name))
                else:
                    # self.current_consumers = [self.workers[consumer_id] for consumer_id in consumer_ids]
                    print("Got the consumers: {0} for node {1} at PE{2}. Transitioning to EXECUTE.".format(consumer_ids, self.current_node.payload, self.name))
                self.state = 'EXECUTE'

            # We know the consumer, they are ready to receive and PE has resources to start execution. Starts execution.
            elif(self.state == 'EXECUTE'):
                yield self.env.timeout(1)
                yield self.env.process(self.execute(self.current_node))
                print("Completed Execution of Node {0} at PE{1}. Transitioning to COMPLETE.".format(self.current_node.payload, self.name))
                self.state = 'COMPLETE'

            # Node has finished execution, let consumer know and release resources.
            elif(self.state == 'COMPLETE'):
                yield self.env.timeout(1)
                self.scheduler.update_completion(self.current_node.name)
                print("Transitioning to POLL at PE{0}".format(self.name))
                self.state = 'POLL'

            elif(self.state == 'DONE'):
                yield self.env.timeout(10)
                break

            else:
                sys.exit("Unknown State: ", self.state)

        print("Completed Execution for PE{0} at {1}".format(self.name, self.env.now))

        return None

    def execute(self, node):
        """Execute the given node"""
        yield self.env.timeout(10)

if __name__ == "__main__":

    from fate.utils.graph import Graph
    from fate.parameters import Parameters
    from fate.DUT.scheduler import Scheduler

    # Create a random Graph.
    testGraph = Graph()
    program  = ['Init', 'GetHandles A', 'GetCoords A', 'GetHandles B', 'GetCoords B', 'Intersect', 'Compute']
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


    env = simpy.Environment()
    parameters = Parameters()
    scheduler = Scheduler(env, parameters)
    update = env.process(scheduler.update_graph(testGraph))
    env.run(update)
    print("Scheduler Instantiated.")

    workers = [ProcessingElement(env, parameters, scheduler, 'Worker '+str(i)) for i in range(5)]
    for worker in workers:
        worker.update_workers(workers)
    print("PEs instantiated.")

    procs = [env.process(worker.run()) for worker in workers] + [env.process(workers[0].check_completion()),]
    proc = env.all_of(procs)
    env.run(proc)