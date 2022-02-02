"""
    This file defines the parameters for the archietcture.
"""
import math

class Parameters:
    """ Represent simulation parameters."""
    def __init__(self):
        ########################################################
        # Pipeline
        ########################################################

        self.NUM_PE = 5
        self.CACHELINE_SIZE = 64
        self.FREQUENCY = 2e9
        self.ADDR_WIDTH = 32
        self.ACCESS_GRANULARITY = 4 # (Access granularity is 32 bits/4 bytes)
        self.ACCESS_GRANULARITY_LOG = math.log(self.ACCESS_GRANULARITY, 2)
        self.CACHELINE_SIZE_LOG = int(math.log(self.CACHELINE_SIZE, 2))
        self.WORDS_PER_LINE = self.CACHELINE_SIZE//self.ACCESS_GRANULARITY
        self.WORDS_PER_LINE_LOG = int(math.log(self.CACHELINE_SIZE//self.ACCESS_GRANULARITY,2))
        self.STORE_QUEUE_SIZE = 16
        self.LOAD_QUEUE_SIZE = 32
        self.L1_READ_PORTS = 2
        ########################################################
        # Instruction Size and code size
        ########################################################
        self.OPCODE_WIDTH = 32
        self.MAX_GRAPH_DEGREE = 256
        self.MAX_GRAPH_NODES = 65536

        ########################################################
        # L1 Cache Control
        ########################################################

        # Data cache
        self.L1D_SIZE = 1024*32
        self.L1D_HIT_LATENCY = 3 # It is effectively 4 cycles, 1 more is added in network
        self.L1D_MISS_LATENCY = 3
        self.L1D_OUTSTANDING_QSIZE = 128
        self.L1D_ASSOCIATIVITY = 8

        # Instruction cache
        self.L1I_SIZE = 1024*64
        self.L1I_OUTSTANDING_QSIZE = 32
        self.L1I_HIT_LATENCY = 0 # Included in the pipeline itself.
        self.L1I_MISS_LATENCY = 4 # Will be accessed from the L2
        self.L1I_ASSOCIATIVITY = 4

        ########################################################
        # L2 Cache Control
        ########################################################

        self.L2_SIZE = 1024*1024
        self.L2_OUTSTANDING_QSIZE = 128
        self.L2_HIT_LATENCY = 7 # It is effectively 8, added 1 in Network.
        self.L2_MISS_LATENCY = 7
        self.L2_ASSOCIATIVITY = 16

        ########################################################
        # SIDECAR Cache Control
        ########################################################

        self.SIDECAR_SIZE = 1024*1024*32
        self.SIDECAR_OUTSTANDING_QSIZE = 128
        self.SIDECAR_HIT_LATENCY = 29 # It is 30, 1 is added in the network
        self.SIDECAR_MISS_LATENCY = 29
        self.SIDECAR_ASSOCIATIVITY = 16

        ########################################################
        # DRAM Control
        ########################################################

        self.DRAM_SIZE = 1024*1024*256 # 1GB
        self.DRAM_ACCESS_LATENCY = 0 # Cycles spent
        self.SIDECAR_DRAM_NOC_LATENCY = 120 
        # Overall 120 cycles for DMC to get the data 
        # (Not including L1-L2-SIDECAR)

        ########################################################
        # NoC Control
        ########################################################

        #Intra PE NoC (Connects PE to L1 caches)
        self.PE_L1_NOC_BW = 1e12 # 1 TB/s
        self.PE_L1_NOC_CACHELINES_PER_CYCLE = self.PE_L1_NOC_BW/(self.FREQUENCY*self.CACHELINE_SIZE)
        self.PE_L1_NOC_LATENCY = 1 # Included in cache access time

        # L1 - L2 NoC (Connects all PEs to shared L2)
        self.L1_L2_NOC_BW = 1e12 # 1 TB/s
        self.L1_L2_NOC_CACHELINES_PER_CYCLE = self.L1_L2_NOC_BW/(self.FREQUENCY*self.CACHELINE_SIZE)
        self.L1_L2_NOC_LATENCY = 1 # Included in cache access time

        # On-Chip NoCs
        self.PE_SIDECAR_NOC_BW = 128e9 # 1 TB/s
        self.PE_SIDECAR_NOC_CACHELINES_PER_CYCLE = self.PE_SIDECAR_NOC_BW/(self.FREQUENCY*self.CACHELINE_SIZE)
        self.PE_SIDECAR_NOC_LATENCY = 1 # Included in cache access time

        # On-Chip NoCs
        self.PE_PE_NOC_BW = 128e9 # 1 TB/s
        self.PE_PE_NOC_CACHELINES_PER_CYCLE = self.PE_PE_NOC_BW/(self.FREQUENCY*self.CACHELINE_SIZE)
        self.PE_PE_NOC_LATENCY = 1 #TODO

        # SIDECAR - DRAM NoC (Connects SIDECAR to DRAM)
        # self.SIDECAR_DRAM_NOC_BW = 64e9 # 64 GB/s
        self.SIDECAR_DRAM_NOC_BW = 140.8e9 # 300 GB/s
        self.SIDECAR_DRAM_NOC_CACHELINES_PER_CYCLE = self.SIDECAR_DRAM_NOC_BW/(self.FREQUENCY*self.CACHELINE_SIZE)

        ########################################################
        # PE Architecture Control
        ########################################################

        # PE Controls
        self.SIMD_WIDTH = 256
        self.REGFILE_SIZE = 32
        self.VECFILE_SIZE = 32
        self.GSU_SIZE = 32
        self.SHARED_LOOPCOUNTER_SIZE = 256
        self.SLS_COMPUTE_LATENCY = 1
        self.NUM_SLS_FUS = 16
        self.MAX_REG_PER_THREAD = 8
        self.MAX_VEC_PER_THREAD = 5
        self.MAX_LOOPS_PER_THREAD = 4
        self.MAX_THREADS = 128
        self.SHARED_REGFILE_SIZE = 16 + self.MAX_REG_PER_THREAD * self.MAX_THREADS
        self.GLOBAL_REGFILE_SIZE = self.REGFILE_SIZE
        self.SHARED_VECFILE_SIZE = 4 + self.MAX_VEC_PER_THREAD * self.MAX_THREADS
        self.GLOBAL_VECFILE_SIZE = self.VECFILE_SIZE
        self.MAX_THREADS_INFLIGHT = 128

        ########################################################
        # Buffet Control
        ########################################################
        self.STR_MEMORY_SIZE = 64
        self.BUFFET_R_LATENCY = 1
        self.BUFFET_W_LATENCY = 1
        self.BUFFET_S_LATENCY = 1

        ########################################################
        # Address Space Control
        ########################################################
        """
        Entire System has 4GB of memory (32 bit Physical addressing)

        Each PE gets 

        64kB of program space 
        32kB of register spill space
        32 kB RESERVED
        TOTAL: 128 kB

        Entire accelerator gets
        64kB of schedule space
        32kB of reg spill space
        32kB RESERVED
        TOTAL: 128 kB

        """
        self.PHYSICAL_MEM_SIZE = 2 ** 32
        self.PE_SPACE = 2 ** 18
        self.PE_PROGRAM_SPACE = 2 ** 16
        self.PE_SPILL_SPACE = 2 ** 15
        self.SYSTEM_SPACE = 2 ** 18
        self.OVERALL_PE_SPACE = self.NUM_PE * self.PE_SPACE

"""
def traversal(visited, sources, root, graph):
    node = graph[root]

    while True:
        # Visited
        if(node.is_source() or node.is_sink()):
            if(first_visit):
                visited[node] = 0
            elif(last_visit):
                visited.remove(node)
            else:
                visited[node] += 1
        # Sources
        if(node.is_source):
            if(first_visit):
                sources.push(node)
            elif(last_visit):
                sources.pop(node)

def traversal(visited, sources, root, graph):
    node = graph[root]
    while flag:
        if(node.is_source()):
            next_node = node.children[visited[node]]
        elif(node.is_sink()):
            if(last_visit and node.have_childrne()):
                next_node = node.children[visited[0]]
            elif(not sources.is_empty()):
                next_node = sources.peek()
            else:
                flag = False
        else:
            if(node.have_children()):
                next_node = node.children[0]
            elif(not source.is_empty()):
                next_node = sources.peek()
            else:
                flag = False
        
        node = next_node

"""