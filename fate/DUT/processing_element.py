import simpy
import sys

from fate.DUT.buffet import Buffet
from fate.DUT.cache import CacheModule
from fate.DUT.network import Network
from fate.DUT.functionalUnit import FunctionalUnit
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

        self.functional_unit = FunctionalUnit(env, parameters, self.l1_dcache)
        self.workers = None
        self.current_node_id = None
        self.current_consumer = None
        self.streaming_memory = {}
        self.producer_handshake_container = simpy.Container(self.env, init=0)
        # Stages of any node: POLL -> STAGE -> EXECUTE -> COMPLETE
        self.state = 'POLL'

    def update_workers(self, workers):
        """Update Workers"""
        self.workers = workers

    def update_streaming_memory(self, node):
        """Based on number of parents, streaming memory gets updated (split)"""
        mem_size = self.parameters.STR_MEMORY_SIZE//max(1, len(node.parents))
        for parent in node.parents:
            self.streaming_memory[parent] =  Buffet(self.env, self.parameters, mem_size)

    def handshake(self):
        """Every Producer must perform a handshake before proceeding. Ensures consumer is ready to receive."""
        yield self.producer_handshake_container.get(1)

    def run(self):
        """
            Main control unit that runs the PE by transitioning through different stages.
        """
        while True:

            if(self.state == 'POLL'):

                # Poll until a node is assigned to the PE from scheduler
                while True:
                    
                    # Ask the scheduler for a node (or timeout if it takes too long)
                    get_node_process = self.env.process(self.scheduler.get_node()) 
                    get_current_node = yield (get_node_process | self.env.timeout(10))

                    # Make sure we did not timeout and _actually_ got a node
                    if(get_node_process in get_current_node):
                        self.current_node = get_current_node[get_node_process]
                        # Looks like the program is complete! Exit the polling
                        if(self.current_node == 'Complete'):
                            print("Program Completion noted in PE{0}. Transitioning to DONE.".format(self.name))
                            self.state = 'DONE'
                            break
                        # Got a valid node. Update required data strucutres and proceed to stage.
                        else:
                            print("Got Node {0} at PE{1}. Transitioning to STAGE.".format(self.current_node.payload, self.name))
                            self.state = 'STAGE'

                            # Udpate the scheduler data structures
                            yield self.env.process(self.scheduler.update_scheduler_by_producer(self.current_node, self.name))
                            # Update local data structures
                            if(len(self.current_node.parents)>0):
                                # Update the streaming memories 
                                self.update_streaming_memory(self.current_node)
                                # Add to container for handshake with producer
                                self.producer_handshake_container.put(len(self.current_node.parents))
                            break

            # Stage the instruction, i.e., remains in staged until it is ready to execute and consumer is identified and ready.
            elif(self.state == 'STAGE'):

                # Step 1: Receive the consumer list from the scheduler
                consumer_ids = yield self.env.process(self.scheduler.get_consumer(self.current_node.name))
                # Step 2: Handshake with every consumer to ensure they are ready.
                if(consumer_ids != None):
                    self.current_consumers = [self.workers[consumer_id] for consumer_id in consumer_ids]
                    for consumer in self.current_consumers:
                        yield self.env.process(consumer.handshake())
                    print("Got the consumers: {0} for node {1} at PE{2}. Transitioning to EXECUTE.".format(consumer_ids, self.current_node.payload, self.name))
                # Terminal Node (No Need to handshake)
                else:
                    self.current_consumers = None
                    print("Terminal Node for node {0} at PE{1}. Transitioning to EXECUTE.".format(self.current_node.payload, self.name))
                # Step 3: Transition to executing the workload
                self.state = 'EXECUTE'

            # We know the consumer, they are ready to receive and PE has resources to start execution. Starts execution.
            elif(self.state == 'EXECUTE'):
                # Execute the current node
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
                yield self.env.timeout(1)
                break

            else:
                sys.exit("Unknown State: ", self.state)

        print("Completed Execution for PE{0} at {1}".format(self.name, self.env.now))

        return None

    def execute(self, node):
        """Execute the given node"""
        yield self.env.timeout(10)
        """
        # Input streams for the node (fixed for any node. For example, Interesect always has 4 in)
        input_streams = list(self.streaming_memory.values())
        # Output streams depend on number of consumers. For each consumer, fixed
        output_consumers = [list(consumer.streaming_memory.values()) for consumer in self.current_consumers]
        yield self.functional_unit.run(node.payload, [])
        """

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