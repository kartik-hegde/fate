"""
    Describes the graph traversal engine. The input program is a Graph data structure (graph.py).
    The input graph is a DFG with instructions as nodes, edges as dependencies.
    This also serves as the core of the PE, which integrates all the major components.

    From a traditional hardware perspective, this unit is fetch/decode+dispatch. Two stages.

    The fetch stage fetches the instructions from the i-cache. The address to fetch is updated by the Dispatch stage.

    The dispatch stage then decodes the fetched instruction, updates some data structures on graph traversal such as
    what nodes were visited etc. Then based on the type of instruction, either it is performed in the same stage (add/sub/loop/if)
    or is dispatched to a functional unit (specialized hardware blocks). Finally, the stage calculates the next address to fetch from.



    NOTES:

    1. In the current formulation, address gen has a dependency on the execution (loop and other controls), this needs to change.
    2. Need to change the master thread storage to level, so that master can be inferred
"""

import simpy
from collections import OrderedDict
import os
import sys
import math
import copy
import random

from fate.DUT.isa import decode
from fate.utils.graph import Node
from fate.DUT.load_store_queue import StoreQueue, LoadQueue

def _print(string):
    pass
    # print(string)

class CurNode:
    def __init__(self, node, payload):
        # Update flags on the type of instruction
        self.name = node.name
        self.is_loop = payload.op_type == 'loop'
        self.is_loop_beg = (payload.instr == 'loop') or (payload.instr == 'while') 
        self.is_loop_end = (payload.instr == 'lend') or (payload.instr == 'whileend')
        self.is_while = payload.instr == 'while'
        self.is_branch = payload.op_type == 'branch'
        self.is_goto = payload.op_type == 'goto'
        self.squash = False
        self.goto = None
        self.goto_level = None
        self.branch_taken = None
        self.is_deterministic = payload.heuristic == 1
        self.is_last_iter = False
        self.cur_node = node
        self.instr = node.payload

        # Node Details (parents, children)
        self.num_parents = max(len(node.parents),1)
        self.num_children = len(node.children)
        self.num_visits_needed = max(self.num_parents+self.num_children-1,1)


        # These will be populated later
        self.node_id = None
        self.was_visited = None
        self.num_visits = None
        self.execute = False

class FATECore:
    """
        This implements an execution of operators set by scheduler. 
    """

    def __init__(self, env, parameters, sls_functional_unit, regfile, vecfile, loopcounter,
                    shared_regfile, shared_vecfile, dcache, icache, logger, access_granularity,
                    gsu, name, smt_width, BASE_ADDR=0):
        self.env = env
        self.parameters = parameters
        self.sls_functional_unit = sls_functional_unit
        self.regfile = regfile
        self.vecfile = vecfile
        self.dcache = dcache
        self.icache = icache
        self.shared_loopcounter = loopcounter
        self.shared_regfile = shared_regfile
        self.shared_vecfile = shared_vecfile
        self.logger = logger
        self.access_granularity_log = int(math.log(access_granularity,2))
        self.gsu = gsu
        self.name = name
        self.issue_width = smt_width
        # Base address of the provided graph
        self.INSTR_BASE_ADDR = BASE_ADDR >> self.access_granularity_log
        self.root = self.INSTR_BASE_ADDR

        # always start at the root. Add pipeline registers.
        self.addr_fetch = [0,] + [None for _ in range(smt_width-1)]
        self.addr_fetch_prev = [-1 for _ in range(smt_width)]

        #### SHARED RESOURCES ACROSS PIPELINES
        # Keyy track the chain you traversed upon. (in case needed to revert.)
        self.visited = dict()# OrderedDict()
        # Keep trace of nodes you need to pop back to
        self.jump = [] #dict()
        self.next_jump = [None for _ in range(smt_width)]
        self.inflight = []  # Store the threads in-flight
        self.l1_read_ports = parameters.L1_READ_PORTS # L1 ports
        self.store_queue = StoreQueue(self.env, parameters.STORE_QUEUE_SIZE) # Load-Store Forwarding
        self.load_queue = LoadQueue(self.env, parameters.LOAD_QUEUE_SIZE) # Load-Store Forwarding
        self.thread_stack = [] # In-flight threads store something once they complete
        self.threads_holding_resource = ['0']
        self.btb = BTB() # {} # Branch Target Buffer
        self.deficit = [0 for _ in range(smt_width)]
        self.cur_time = 0
        self.words_per_line = parameters.WORDS_PER_LINE
        self.cacheline_width = parameters.CACHELINE_SIZE
        self.active_loads = 0
        self.pending_stores = 0
        self.btb_logger = [["Last Addr", "Next Address", "Predicted Address"],]
        # self.addr_aligned_bits = int(math.log(parameters.WORDS_PER_LINE,2))
        # self.addr_aligned_mask = (2 ** self.addr_aligned_bits)-1

        # Current Node and its properties
        self.cur_node = [None for _ in range(smt_width)]
        self.num_visits_needed = 0
        self.num_visits = 0

        # Extra resources to enable OoO
        self.threadID = ['0',] + [None for _ in range(smt_width-1)]
        self.master_strand = ['0',] + [None for _ in range(smt_width-1)]
        self.unique_id = 1
        self.flag = False

        # Flags
        self.active = [True,] + [False for _ in range(smt_width-1)]
        self.stalled = [False for _ in range(smt_width)]
        self.done = [False for _ in range(smt_width)]

    def step(self):
        """
            This runs one step of pipeline every cycle.
        """

        for issue_slot in range(self.issue_width):

            # Flag flags
            done = False

            # Local variables for the current Slot information
            threadID = self.threadID[issue_slot]
            master_strand = self.master_strand[issue_slot]
            # Fetch Address for the SMT Slot
            fetch_addr = self.addr_fetch[issue_slot]

            # If there is a deficit (for e.g., due to a branch mistake, a bubble is added)
            if(self.deficit[issue_slot]):
                self.deficit[issue_slot] -= 1
                continue

            ########## 1. Instruction Fetch ################

            if(fetch_addr != None):
                # Add the base address
                fetch_addr = self.INSTR_BASE_ADDR + (fetch_addr << self.access_granularity_log)
                # print("\n\n-- FETCH stage with address {0} at time {1}".format(self.addr_fetch, self.env.now))
    
                # Fetch from the graph.
                fetched_node = yield self.env.process(self.icache.read(fetch_addr))
                cur_node = copy.deepcopy(fetched_node)
                # print("Successfully fetched {0} (Addr: {1}, PE: {2}, Slot: {3}\n".format(cur_node.__dict__, fetch_addr, self.name, issue_slot))
            else:
                cur_node = None
            

            # There are three steps: decode, execute, and addr_gen
            # Decode and execute are performed only if there is a valid fetched node
            # Addr gen is impacted if the core is in wait state (for threads to resume) or active (valid fethed node)

            ########## 2 and 3. DECODE and DISPATCH ################


            if(cur_node is not None):    

                # First, decode the node.
                payload = decode(cur_node) # self.cur_node.payload
                # Get details of the current node
                node = CurNode(cur_node, payload)

                # print("\n\n -------------------------------------------------------------------------------------------- \n\n")
                # print("\n\n INSTRUCTION : {0} at time {1} and thread {2} in PE {3}: slot {4}".format(node.instr, self.env.now, threadID, self.name, issue_slot))
                # input()
                # print("\nDISPATCH stage. Thread state active:{0}, was any data fetched:{1} in PE {2}".format(self.active[issue_slot],self.cur_node is not None, self.name))
                # print("ENVIRONMENT Active processes {0} and queue {1} at {2}".format(self.env._queue, self.env._active_proc, self.env.now))

                # Next Jump read (this is used for branch prediction)
                self.next_jump[issue_slot] = self.thread_stack[0][-1] if(self.thread_stack) else (int(self.jump[-1].split("_")[0]) if(self.jump) else None)
                # self.next_jump[issue_slot] = int(self.jump[-1].split("_")[0]) if(self.jump) else None

                # Logger Functions
                if(self.env.now - self.cur_time > 5000):
                    self.cur_time = self.env.now
                    # self.get_memory_consumption()
                    # self.get_datastructure_size()
                    self.update_logger(payload)
                    # print(self.resource_usage())
                    # input()

                # To know how many times this node was visited before
                # LOG2(MAX_NODES) + LOG2(MAX_STRANDS)
                node.node_id = str(node.name) + '_' + master_strand
                node.was_visited = node.node_id in self.visited
                node.num_visits = self.visited[node.node_id]+1 if(node.was_visited) else 1

                # Execute the node in two cases:
                # 1. If the node has more than 1 parent, then it is executed only after both parents complete
                # 2. If the node has more than 1 children, execute only the first visit (TODO: Case where multiple executions needed)
                # Both cases are captured if you execute only when your visit is Nth, whenre N is the number of parents.
                # Note that a special case is loops. We visit the loop to make sure the num visits are up to date.
                node.execute = True if((node.num_visits == node.num_parents) or node.is_loop) else False
                # print("DEBUG This instruction will be executed: ", execute)
                # print("DEBUG Instruction is Long Latency: ", not self.is_deterministic)

                ########## 3. EXECUTE/DISPATCH ################


                # Note the time and cycles (perf counter)
                start_time = self.env.now
                compute_cycles = 1

                # If the heuristic is deterministic, then we execute right away
                if(node.execute and node.is_deterministic):

                    # print("DEBUG Executing a deterministic operator {0}".format(payload.instr))
                    if(payload.op_type == 'loop'):

                        # 1. For Loop (LOOP <NUM_ITERS> <ADDR_REG> <LOOP END>)
                        if(payload.instr == 'loop'):
                            # This node needs to be visited these many times or at least once
                            loop_count = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            # If the loop count is 0 or lower, no work: becomes a go-to to end of the loop
                            if(loop_count<1):
                                node.goto = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            else:
                                # Keep the loop counter updated
                                count = yield self.env.process(self.read_operand(payload.operands[1], threadID, increment=True))
                                # We will update number of children for this node as the number of iterations.
                                node.num_children = loop_count
                                node.num_visits_needed = max(node.num_parents+node.num_children-1,1)

                                # BFS Keep track
                                # if(('loop GR7 LC0.' in node.instr) or ('loop GR5 LC0.' in node.instr)):
                                #     if((loop_count-count)%1000==0):
                                #         print("Completed {0}/{1} of outer loop. PE {2}\n".format(count,loop_count, self.name))
                                # elif('loop GR5 LC1.' in node.instr):
                                #     if((loop_count-count)%1000==0):
                                #         print("Completed {0}/{1} of outer loop. PE {2}\n".format(count,loop_count, self.name))
                                # if('loop GR10 LC0.' in node.instr):
                                #     if((loop_count-count)%500==0):
                                #         print("Completed {0}/{1} of inner loop. PE {2}\n".format(count,loop_count, self.name))
                                # print("LOOP, Current count: {0}, Loop bound: {1}".format(count, loop_count))

                        # While Loop (WHILE <NUM_ITERS> <LOOP START> <LOOP END>)
                        elif(payload.instr == 'while'):

                            # This is used to get the address to jump to 
                            # (These preceding instructions set the flag used to determine the end condition)
                            node.while_start = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            
                            # If the flag is true, loop continues, else set the loop bound to num visits and go to loop end.
                            if(self.flag):
                                node.num_children = math.inf
                            else:
                                # Go to end
                                node.goto = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                node.num_children = node.num_visits-node.num_parents+1
                                yield self.env.process(self.write_operand(payload.operands[0], node.num_visits, threadID))

                            node.num_visits_needed = max(node.num_parents+node.num_children-1,1)

                        # End of a loop (LOOPEND <NUM_ITERS> <ADDR_REG>)
                        elif(payload.instr == 'lend'): 
                            # We will consider number of parents as the loop iterations needed
                            loop_count = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            count = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            node.num_parents = loop_count
                            node.num_visits_needed = max(node.num_parents+node.num_children-1,1)
                            # print("LOOP END, Current count: {0}, Loop bound: {1}".format(count, loop_count))

                        # While Loop End (WHILEEND <NUM_ITERS> <ADDR_REG>)
                        elif(payload.instr == 'whileend'):
                            # We will update number of children for this node as the number of iterations. 
                            # (This is set to a large value until the while loop flag is unset)
                            node.num_parents = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            node.num_visits_needed = max(node.num_parents+node.num_children-1,1)

                    # 2. Register operations (these are performed right here, and not sent to accelerator FUs)
                    elif(payload.op_type == 'reg'):
                        # Add
                        if(payload.instr == 'add'):
                            # result = src0 + src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 + src1
                            # print("ADD", result,src0,src1)
                            # if(node.instr == 'add GR9 GR9 1'):
                            #     print("Working on Component {0}".format(result))
                            # elif(node.instr == 'add GR14 GR14 1'):
                            #     print("Working on Node {0}".format(result))
                            # if(self.cur_node.payload == 'add GR9 GR9 1'):
                                # print("Level Updated to {0}".format(result))
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Add
                        elif(payload.instr == 'addi'):
                            # result = src0 + src1 << 2
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 + (src1 << 2)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Subtract
                        elif(payload.instr == 'sub'):
                            # result = src0 - src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 - src1
                            # print("SUB:", result, src0, src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # MAC
                        elif(payload.instr == 'mac'):
                            # result = src0 - src1
                            src = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src + (src0 * src1)
                            # print("SUB:", result, src0, src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Abs Subtract
                        elif(payload.instr == 'subf'):
                            # result = src0 - src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = abs(src0 - src1)
                            # print("SUB:", result, src0, src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Ceil
                        elif(payload.instr == 'ceil'):
                            # result = src0 - src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = int(math.ceil(float(src0) / src1))
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Max
                        elif((payload.instr == 'max') or (payload.instr == 'min')):
                            # result = src0 - src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = max(src0,src1) if(payload.instr == 'max') else min(src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Multiply
                        elif(payload.instr == 'mul'):
                            # result = src0 * src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 * src1
                            # print("MUL", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Division
                        elif(payload.instr == 'div'):
                            # result = src0 * src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 / src1
                            # print("DIV", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Move
                        elif(payload.instr == 'mov'):
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # print("MOV:", src0)
                            yield self.env.process(self.write_operand(payload.operands[0],src0, threadID))

                        # Move
                        elif(payload.instr == 'clear'):
                            # print("Clear:")
                            yield self.env.process(self.write_operand(payload.operands[0],-1, threadID))

                        # Swap
                        elif(payload.instr == 'swp'):
                            src0 = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # print("SWP:", src0, src1, self.name)
                            yield self.env.process(self.write_operand(payload.operands[0],src1, threadID))
                            yield self.env.process(self.write_operand(payload.operands[1],src0, threadID))

                        # Vector Move
                        elif(payload.instr == 'vmov'):
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # print("VMOV:", src0)
                            yield self.env.process(self.write_operand(payload.operands[0],[src0]*self.parameters.VECTOR_WIDTH, threadID))

                        # Move
                        elif(payload.instr == 'lsl'):
                            # result = src0 << src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 << src1
                            # print("LSL", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Aligned Add <dest><align src0><src1>
                        elif(payload.instr == 'addaligned'):
                            # result = src0 << src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 - src0%self.words_per_line + src1
                            # print("Add Aligned", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Shift Add <dest><align src0><src1<<2>
                        elif(payload.instr == 'addsi'):
                            # result = src0 << src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0 + (src1<<2)
                            # print("Add Aligned", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Index into Vector <dest><vector><index>
                        elif(payload.instr == 'movvr'):
                            # result = src0 << src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            result = src0[src1%self.words_per_line]
                            # print("MOVVR", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                        # Index into Vector <dest><src0><src1>
                        elif(payload.instr == 'cswp'):
                            # result = src0 << src1
                            dest = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            if(src0 == src1):
                                yield self.env.process(self.write_operand(payload.operands[2],dest, threadID))
                                self.stalled[issue_slot] = False
                            else:
                                self.stalled[issue_slot] = True

                        else:
                            sys.exit("Reg-Reg Instruction {0} not understood.".format(payload.instr))

                    # 3. Vector operations
                    elif(payload.op_type == 'reg'):
                        # Vector Add
                        if(payload.instr == 'vadd'):
                            # result = src0 + src1
                            src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            assert (type(src0)==list) and (type(src1)==list), 'Received {0} and {2} of type {1} and {3}'.format(src0, type(src0), src1, type(src1))
                            result = [src0[idx] + src1[idx] for idx in range(len(src0))]
                            # print("VADD", result,src0,src1)
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                    # 4. Branch Operations
                    elif(payload.op_type == 'branch'):
                        dest = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        src0 = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        src1 = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                        # BEQ <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        if(payload.instr == 'beq'):
                            if(src0 == src1):
                                node.branch_taken = dest

                        # BNEQ <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bneq'):
                            if(src0 != src1):
                                node.branch_taken = dest
                            # print("BNEQ", dest, src0!=src1)

                        # BG <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bg'):
                            if(src0 > src1):
                                node.branch_taken = dest
                            # print("BG", dest, src04!=src1)
                            if('bg 48 LR9 GR7' in node.instr):
                                if((src0)%1280==0):
                                    print("Completed {0}/{1} of inner loop. PE {2}\n".format(src0,src1, self.name))


                        # BG <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bng'):
                            if(src0 <= src1):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        # BL <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bl'):
                            if(src0 < src1):
                                node.branch_taken = dest

                       # BNL <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bnl'):
                            if(src0 >= src1):
                                node.branch_taken = dest

                        # BG <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'bge'):
                            if(src0 >= src1):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        # BLE <DEST> <SRC0> <SRC1> (go to dest if src0==src1)
                        elif(payload.instr == 'ble'):
                            if(src0 <= src1):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        # BAL <DEST> <SRC0>  (go to dest if src0 is aligned)
                        elif(payload.instr == 'bal'):
                            if(src0 % src1 == 0):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        # BUNAL <DEST> <SRC0>  (go to dest if src0 is unaligned)
                        elif(payload.instr == 'bunal'):
                            if(src0 % src1 != 0):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        # BALR <DEST> <SRC0>  (go to dest if src0+1 is aligned)
                        elif(payload.instr == 'balr'):
                            if((src0+1) % src1 == 0):
                                node.branch_taken = dest
                            # print("BG", dest, src0!=src1)

                        else:
                            sys.exit("Instruction Branch {0} not understood.".format(payload.instr))

                    # 4. Conditional Operations
                    elif(payload.op_type == 'cond'):
                        # Read operands
                        src0 = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        src1 = yield self.env.process(self.read_operand(payload.operands[1], threadID))

                        # CMP <SRC0> <SRC1> (Compare)
                        if(payload.instr == 'cmp'):
                            self.flag = src0 == src1

                        # GT <SRC0> <SRC1> (Greater Than)
                        elif(payload.instr == 'gt'):
                            self.flag = src0 > src1

                        # GE <SRC0> <SRC1> (Greater Than/Equal to)
                        elif(payload.instr == 'ge'):
                            self.flag = src0 >= src1

                        # LT <SRC0> <SRC1> (Less Than)
                        elif(payload.instr == 'lt'):
                            self.flag = src0 < src1

                        # LE <SRC0> <SRC1> (Less Than/Equal to)
                        elif(payload.instr == 'le'):
                            self.flag = src0 <= src1

                        else:
                            sys.exit("Instruction Conditional {0} not understood.".format(payload.instr))

                    # 4. Sync operations. These interface to the GSU.
                    elif(payload.op_type == 'sync'):

                        # Poll
                        if(payload.instr == 'poll'):
                            time = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            yield self.env.process(self.gsu.poll(payload.operands[0].operand_value, time))
                        # Poll
                        elif(payload.instr == 'acquire'):
                            lock = yield self.env.process(self.gsu.acquire(payload.operands[1].operand_value))
                            yield self.env.process(self.write_operand(payload.operands[0],lock, threadID))
                        # Poll
                        elif(payload.instr == 'release'):
                            yield self.env.process(self.gsu.release(payload.operands[1].operand_value))
                        # Read
                        elif(payload.instr == 'aread'):
                            data = yield self.env.process(self.gsu.read(payload.operands[1].operand_value))
                            yield self.env.process(self.write_operand(payload.operands[0],data, threadID))
                        # Update (write)
                        elif(payload.instr == 'aupdate'):
                            data = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            yield self.env.process(self.gsu.update(payload.operands[0].operand_value, data))
                        # Atomic Increment
                        elif(payload.instr == 'aincr'):
                            if(len(payload.operands)>2):
                                incr_val = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                data = yield self.env.process(self.gsu.incr(payload.operands[1].operand_value, incr_val))
                            else:
                                data = yield self.env.process(self.gsu.incr(payload.operands[1].operand_value))
                            yield self.env.process(self.write_operand(payload.operands[0],data, threadID))
                            if(node.instr == 'aincr GR20 AR3'):
                               print("Got Spot {0} at PE {1}".format(data, self.name))
                            # elif(node.instr == 'aincr LR14 AR1'):
                            #    print("Grabbed {0} at PE {1}".format(data, self.name))
                            # elif(node.instr == 'aincr GR9 AR4'):
                            #    print("\n\n-------- Level up {0} at PE {1} ------- \n\n".format(data, self.name))
                        # Wait
                        elif(payload.instr == 'wait'):
                            time_to_wait = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                            yield self.env.timeout(time_to_wait)

                        # Barrier: Wait till all strands complete. TODO
                        elif(payload.instr == 'barrier'):
                            while((not self.store_queue.empty()) or (self.pending_stores != 0)):
                                yield self.env.timeout(1)
                            # print(" -- BARRIER -- ", self.pending_stores)

                    elif(payload.instr == 'goto'):
                        dest = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        node.goto = dest
                        if(len(payload.operands)>1):
                            node.goto_level = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        else:
                            node.goto_level = None
                        # print("GOTO {0} at PE {1}".format(dest, self.name))

                    # Specialized Instructions
                    elif(payload.op_type == 'special'):

                        # unvisited <dest> <vector> <val>
                        if(payload.instr == 'unvisited'):
                            # Address to load
                            vector = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            val = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            # Index of first val
                            if(val in vector):
                                idx_val = vector.index(val)
                                # Update it to 0
                                vector[idx_val] = 0
                            else:
                                idx_val = self.words_per_line
                            # Update vector
                            # yield self.env.process(self.write_operand(payload.operands[1],vector, threadID)) 
                            # Update the index
                            yield self.env.process(self.write_operand(payload.operands[0],idx_val, threadID))

                        # neighbors <num_neighbors> <base> <vector>
                        if(payload.instr == 'neighbors'):
                            # Address to load
                            vector = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            val = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            # Align the value
                            val = val%self.words_per_line
                            result = vector[val+1]-vector[val]
                            # print("neighbors", result, vector, val)
                            # Update the Result
                            yield self.env.process(self.write_operand(payload.operands[0],result, threadID))

                    # Blocked Load operation LOAD <DEST REG> <ADDR>
                    elif(payload.op_type == 'bload'):

                        # Blocked Vector Load
                        if((payload.instr == 'vbload') or (payload.instr == 'vbloadi')):
                            # Address to load
                            load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # If another operand is supplied, then increment is performed.
                            if(len(payload.operands)>2):
                                load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                load_addr += load_addr_incr<< (int(self.parameters.CACHELINE_SIZE_LOG) if(payload.instr == 'vbloadi') else 2)

                            # Address has to be aligned with the line
                            load_addr = load_addr - load_addr%self.cacheline_width

                            # Read the value from D-cache
                            read_data = yield self.env.process(self.dcache.read(load_addr, line_read=True))
                            # print("BVLOAD: Read Vector: {0} for address {1}".format(read_data, load_addr))
                            # Write to the destination
                            yield self.env.process(self.write_operand(payload.operands[0],read_data, threadID))

                        # Blocked Vector Load
                        elif((payload.instr == 'dvbload') or (payload.instr == 'dvbloadi')):
                            # Address to load
                            load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # If another operand is supplied, then increment is performed.
                            if(len(payload.operands)>2):
                                load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                load_addr += load_addr_incr<< (int(self.parameters.CACHELINE_SIZE_LOG) if(payload.instr == 'dvbloadi') else 2)

                            # Address has to be aligned with the line
                            load_addr = load_addr - load_addr%self.cacheline_width

                            # Read the value from D-cache
                            read_data = yield self.env.process(self.dcache.deep_read(load_addr, line_read=True))
                            # print("BVLOAD: Read Vector: {0} for address {1}".format(read_data, load_addr))
                            # Write to the destination
                            yield self.env.process(self.write_operand(payload.operands[0],read_data, threadID))

                        # Blocked Scalar Load
                        elif((payload.instr == 'bload') or (payload.instr == 'bloadi')):
                            # Address to load
                            load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # If another operand is supplied, then increment is performed.
                            if(len(payload.operands)>2):
                                load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                load_addr += load_addr_incr if(payload.instr == 'bload') else load_addr_incr<<2

                            read_data = yield self.env.process(self.dcache.read(load_addr))

                            # if(node.instr == 'bloadi LR1 GR5 LC0.'):
                            #     print("Loading from Addr {0} {1}".format(load_addr, read_data))
                            #     print(" \n\n ------ Node {0} --------- ".format(read_data))
                            # elif(node.instr == 'dbloadi LR6 GR3 LR1'):
                            #     print("Visited : {0} --------- ".format(read_data))
                            # if(self.name == 0):
                                # print("BLOAD: Read data {0} for address {1} ({3}) in PE{2}".format(read_data, load_addr, self.name,load_addr_incr))
                            # Write to the destination
                            yield self.env.process(self.write_operand(payload.operands[0],read_data, threadID))

                        # Blocked Scalar Deep Load
                        elif((payload.instr == 'dbload') or (payload.instr == 'dbloadi')):
                            # Address to load
                            load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                            # If another operand is supplied, then increment is performed.
                            if(len(payload.operands)>2):
                                load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                                load_addr += load_addr_incr if(payload.instr == 'dbload') else load_addr_incr<<2

                            # Read the value from D-cache
                            read_data = yield self.env.process(self.dcache.deep_read(load_addr))

                            # if(node.instr == 'bloadi LR1 GR5 LC0.'):
                            #     print("Loading from Addr {0} {1}".format(load_addr, read_data))
                            #     print(" \n\n ------ Node {0} --------- ".format(read_data))
                            # elif(node.instr == 'dbloadi LR6 GR3 LR1'):
                            #     print("Visited : {0} --------- ".format(read_data))
                            # if(self.name == 0):
                            #     print("BLOAD: Read data {0} for address {1} in PE{2}".format(read_data, load_addr, self.name))
                            # Write to the destination
                            yield self.env.process(self.write_operand(payload.operands[0],read_data, threadID))

                        else:
                            sys.exit("Blocked Load insutrction {0} not understood.".format(payload.instr))

                    # Store operation STORE <DEST REG> <ADDR>
                    elif(payload.op_type == 'store' or payload.op_type == 'storei'):
                        # Address to store
                        store_addr = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        # If another operand is supplied, then increment is performed.
                        if(len(payload.operands)>2):
                            store_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            store_addr += (store_addr_incr if(payload.op_type == 'store') else store_addr_incr<<2)                    
                        # Store data
                        store_data = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        # Add to pending stores
                        self.pending_stores += 1

                        # if(node.instr == 'storei GR6 LR5 GR10'):
                        #     print("Storing to {0} {1}".format(store_addr, store_data))
                            # print("adding {0}".format(store_data))
                        # Add the store to the store queue
                        # print("Storing to {0} {1}".format(store_addr, store_data))
                        yield self.env.process(self.store_queue.enqueue([store_addr, store_data, False, False]))

                    elif(payload.op_type == 'dstore' or payload.op_type == 'dstorei'):
                        # Address to store
                        store_addr = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        # If another operand is supplied, then increment is performed.
                        if(len(payload.operands)>2):
                            store_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            store_addr += (store_addr_incr if(payload.op_type == 'dstore') else store_addr_incr<<2)                    
                        # Store data
                        store_data = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        # Add to pending stores
                        self.pending_stores += 1

                        # if(node.instr == 'storei GR6 LR5 GR10'):
                        #     print("Storing to {0} {1}".format(store_addr, store_data))
                            # print("adding {0}".format(store_data))
                        # Add the store to the store queue
                        # print("Storing to {0} {1}".format(store_addr, store_data))
                        yield self.env.process(self.store_queue.enqueue([store_addr, store_data, False, True]))

                    # Load operation LOAD <DEST REG> <ADDR>
                    elif(payload.op_type == 'vstore'):
                        # Address to store
                        store_addr = yield self.env.process(self.read_operand(payload.operands[0], threadID))
                        # Store data
                        store_data = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        # Offset to the store data
                        if(len(payload.operands)>2):
                            store_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            store_addr += (store_addr_incr << int(self.parameters.CACHELINE_SIZE_LOG))

                        # Add the store to the store queue
                        yield self.env.process(self.store_queue.enqueue([store_addr, store_data, True, False]))

                    elif(payload.op_type == 'nop'):
                        # DO Nothing
                        pass

                    else:
                        sys.exit("Instruction Not Understood: {0}".format(payload.op_type))


                    ########## 3. NEXT ADDR GEN ################
                    # Done: Whether sim is complete, next_addr_fetch: The next address to be fetched
                    # ThreadID: threadID for the next step, next_addr_thread: in case of a ND node, this is the next addr after resuming.
                    done, next_addr_fetch, threadID_next, next_addr_thread, master_strand_next = self.addr_gen(node, threadID, master_strand, issue_slot)

                # If the heuristic is non-deterministic, we launch a process and move on.
                # We come back and check if it has finished.
                elif(node.execute and (not node.is_deterministic)):

                    ########## 3. NEXT ADDR GEN ################
                    # Done: Whether sim is complete, next_addr_fetch: The next address to be fetched
                    # ThreadID: threadID for the next step, next_addr_thread: in case of a ND node, this is the next addr after resuming.
                    done, next_addr_fetch, threadID_next, next_addr_thread, master_strand_next = self.addr_gen(node, threadID, master_strand, issue_slot)

                    # Create the an independent process (It writes to the thread stack once complete)
                    # Load operation LOAD <DEST REG> <ADDR><INCR>
                    if(payload.op_type == 'load'):

                        # Address to load
                        load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        # If another operand is supplied, then increment is performed.
                        if(len(payload.operands)>2):
                            load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            load_addr += (load_addr_incr if(payload.instr == 'load' or payload.instr == 'dload') else load_addr_incr<<2)

                        # Slice the address (to get the cacheline and the offset)
                        # load_addr_aligned, load_offset = load_addr>>self.addr_aligned_bits, load_addr&self.addr_aligned_mask
                        # print("Load Addr Base: {0}, Increment: {1}".format(load_addr, load_addr_incr))

                        # Option 1: Check the load-store queue for forwarding, memory consistency
                        store_queue_entry = self.store_queue.check(load_addr)
                        if(store_queue_entry != None):
                            read_data = store_queue_entry
                            # Write to the destination
                            yield self.env.process(self.write_operand(payload.operands[0],read_data, threadID))
                            # Need to notify that the work is complete.
                            self.thread_stack.append((threadID, master_strand, next_addr_thread))                        
                        # elif(load_addr_aligned in self.load_buffer):
                        #     read_data = self.load_buffer[load_addr_aligned][load_offset]
                        #     yield self.env.timeout(1)
                        # Option 2: Check if it was recently accessed, if yes, update it. TODO
                        # Otherwise, read the value from L1D-cache
                        else:
                            # Prepare load packet
                            deep_read = True if(payload.instr == 'dload' or payload.instr == 'dloadi') else False
                            line_read = False
                            load_packet = [load_addr, line_read, deep_read, payload.operands[0], threadID, master_strand, next_addr_thread]
                            # Append to load queue
                            self.active_loads += 1
                            yield self.env.process(self.load_queue.enqueue(load_packet))


                    # Load operation VLOAD <DEST REG> <ADDR><INCR>
                    elif(payload.op_type == 'vload'):

                        # Address to load
                        load_addr = yield self.env.process(self.read_operand(payload.operands[1], threadID))
                        # If another operand is supplied, then increment is performed.
                        if(len(payload.operands)>2):
                            load_addr_incr = yield self.env.process(self.read_operand(payload.operands[2], threadID))
                            load_addr += load_addr_incr << (int(self.parameters.CACHELINE_SIZE_LOG) if(payload.instr=='vloadi') else 2)

                        # Address has to be aligned with the line
                        load_addr = load_addr - load_addr%self.cacheline_width

                        # Prepare load packet
                        deep_read = True if(payload.instr == 'dvload' or payload.instr == 'dvloadi') else False
                        line_read = True
                        load_packet = [load_addr, line_read, deep_read, payload.operands[0], threadID, master_strand, next_addr_thread]
                        # Append to load queue
                        yield self.env.process(self.load_queue.enqueue(load_packet))            
                    else:
                        sys.exit("Non-deterministic Instruction Not Understood {0}".format(payload.op_type))                    

                # We are not executing anything
                else:
                    # Retain the same threadID
                    done, next_addr_fetch, threadID_next, next_addr_thread, master_strand_next = self.addr_gen(node, threadID, master_strand, issue_slot)

                # self.serialize[identifier].release(lock)
                # self.serialize.remove(identifier)

            # Thread is in dormant state (Not active)
            else:
                start_time_idle = self.env.now
                # print("DEBUG Waiting for threads to finish")
                # No node was dispacthed
                node = None
                # Skip the time till one of the threads write something to the thread_stack
                # input()
                # Retain the same threadID
                done, next_addr_fetch, threadID_next, next_addr_thread, master_strand_next = self.addr_gen(node, threadID, master_strand, issue_slot)

                # This is the only part of the engine where the engine is not doing anything.
                self.logger['idle'] += self.env.now - start_time_idle

            # Update next address and operator
            self.threadID[issue_slot] = threadID_next
            self.master_strand[issue_slot] = master_strand_next
            self.addr_fetch[issue_slot] = next_addr_fetch

            # Check if there is a branch miss, if yes, it adds a deficit
            if(node is not None):
                deficit = self.branch_miss(issue_slot, node.is_deterministic, node.is_loop_end)  
            else:
                deficit = 0

            self.deficit[issue_slot] += deficit
            # Only master slot decides if the simulation is complete 
            # if(master_slot):
            self.done[issue_slot] = done

        return None

    def load_execute(self, packet):
        """
            Independent process that completes the load.
        """
        # Extract the packet
        load_addr, line_read, deep_read, dest, threadID, master_strand, next_addr_thread = packet

        # Now read from L1D
        start_time = self.env.now
        if(deep_read):
            read_data = yield self.env.process(self.dcache.deep_read(load_addr, line_read=line_read))
        else:
            read_data = yield self.env.process(self.dcache.read(load_addr, line_read=line_read))
        self.logger['access_latency'].append(self.env.now-start_time)

        # Write to the destination
        yield self.env.process(self.write_operand(dest,read_data, threadID))
        # Need to notify that the work is complete.
        self.thread_stack.append((threadID, master_strand, next_addr_thread))
        # Active load completes
        self.active_loads -= 1

    def store_execute(self, addr, data, line_write, deep_store):
        """
            Ensures that a store is fully complete until the lowest level.
        """
        if(deep_store):
            yield self.env.process(self.dcache.deep_write(addr, data, line_write=line_write))
        else:
            yield self.env.process(self.dcache.write(addr, data, line_write=line_write))
        self.pending_stores -= 1
        # yield simpy.events.AnyOf(self.env, list(self.proc_inflight.values()))

    def store_launch(self):
        """
            Independent process that maintains the store queue. 
        """
        # Runs until the end of simulation
        while(not all(self.done)):
            # Check if anything exists in Store queue.
            if(not self.store_queue.empty()):
                addr, data, line_write, deep_store = self.store_queue.dequeue()
                # Launch the store: Assume 1 L1 Store Slot
                self.env.process(self.store_execute(addr, data, line_write, deep_store))

            # Tick the clock
            yield self.env.timeout(1)

        return None

    def load_launch(self):
        """
            Independent process that maintains the load queue. 
        """
        # Runs until the end of simulation
        while(not all(self.done)):
            # As many dequeues/cycle as the number of Load Ports
            for _ in range(self.l1_read_ports):
                # Read the queue
                packet = self.load_queue.dequeue()
                # Check if the read is valid, if yes, send it out
                if(packet != None):
                    self.env.process(self.load_execute(packet))
            # Tick the clock
            yield self.env.timeout(1)

        return None


    def addr_gen(self, node, threadID, master_strand, issue_slot):
        """
            This function is responsible for updating the next address to fetch from.
            This happens in a cycle in parallel to the dispatch, and a delay is not attached.
        """

        ### This has two cases.
        ### There is an active thread running, hence there was an actual node fetched.
        ### Or, there is no active thread, but there are dormant threads, in which case, we wait for them to become active.
        done = False
        master_slot = issue_slot==0
        
        if(self.active[issue_slot] and not self.stalled[issue_slot]):
            # Defaults     

            threadID_next = threadID
            master_strand_next = master_strand

            # Get the unique name of the node
            # node_id = str(node.name) + '_' + master_strand

            # Setup useful Flags about the node
            was_visited = node.node_id in self.visited
            node.was_visited = node.node_id in self.visited
            node.num_visits = self.visited[node.node_id]+1 if(node.was_visited) else 1

            is_last_visit = (node.num_visits == node.num_visits_needed)
            executed = (node.num_visits >= node.num_parents)
            # Check if this node exists in the visited List
            # Find a better logic for Sink Node TODO
            is_sink_node = (node.num_parents > 1) or node.is_loop_end # (not is_jump_entry_present) and node.was_visited
            is_source_node = (node.num_children > 1) or node.is_loop_beg
            source_node_first_visit = is_source_node and (not was_visited)

            ### THE BELOW CONTROL LINES SHOULD TURN INTO ALWAYS BLCOKS ###

            #### Update Jump back list ####
            # Delete the entry if we are not going to visit again
            if(node.was_visited and is_last_visit and is_source_node):
                # del self.jump[-1]
                # TODO: This is going to break for While loop, fix it.
                self.jump.remove(str(node.name) + '_' + master_strand + '_1')
            # Create an entry only if there are more than 1 children (need to revisit)
            elif((not was_visited) and (not is_last_visit) and (node.num_visits_needed>1) and (node.num_children>1) and (node.num_visits>= node.num_parents)):
                # To determine the Strand ID for the jump back, we store the master_strand
                if(node.is_while):
                    # In case of a while loop, we need to jump to the condition checking preface
                    # The suffix 0 refers to the fact that the jump back will need to retain the threadID (not a source)
                    jump_id = str(node.while_start) + '_' + master_strand + '_0'
                else:
                    jump_id = str(node.name) + '_' + threadID + '_1'

                # print("Updated master thread to ", master_strand)
                self.jump.append(jump_id)
                

            # Update node ID (We assume the source node belongs to the next master_strand)
            if(is_source_node and (not was_visited)):
                node.node_id = str(node.name) + '_' + (master_strand if(node.is_while) else threadID)

            #### Update visitor's list ####

            # Delete the entry if we are not going to visit again
            if(node.was_visited and is_last_visit):
                # if the only entry for the node is the master thread, remove the entire entry.
                del self.visited[node.node_id]
            # Create an entry only if number of visits needed is more than 1
            elif((not node.was_visited) and (not is_last_visit) and (node.num_visits_needed>1)):
                # Create a visited entry
                self.visited[node.node_id] = 1
            #Key present, increment
            elif((was_visited) and (not is_last_visit)):
                self.visited[node.node_id] += 1

            ###### Accommodate thread switching #######
            # Create Flags

            # There are 4 outputs: whether this is the last step (done), what's the next address to fetch, what's the next threadID, if we are switching then what is the address
            # to go to when this thread is resumed.

            # We will resume a thread in 2 cases: either you hit a ND node, or current thread is complete

            # We either go down the graph, or jump back to another location (new strand or resume strand)

            # Condition 1: When do we continue down the graph (depth)
            # There should be a children, the node must've been executed (all parents visited), node must be deterministic, and should not be a go to.
            keep_going_down = (node.num_children > 0) and executed and node.is_deterministic and (not node.is_goto) and (not node.squash)

            # Condition 2: When do we want to jump elsewhere
            # Of course, keep_going_down must be False, and one of these must be true: end of a chain (no children), was not executed, not deterministic, along with
            # that there should be a place to jump to: thread stack (a completed strand), self.jump (source node), 
            jump_back       = not keep_going_down \
                                and ((is_last_visit and (node.num_children==0)) or (not executed) or node.is_goto or (not node.is_deterministic)) \
                                    and (self.thread_stack or self.jump or node.is_goto) and (not node.squash)

            # Different choices of jump back (not going down)
            go_to_addr      = jump_back and node.is_goto
            resume_thread   = jump_back and (self.thread_stack) and (not node.is_goto)
            create_thread   = jump_back and ((not self.thread_stack) and self.jump and (not keep_going_down) and (self.are_resources_available())) \
                                and (not node.is_goto) and (master_slot)

            # Condition 4 There are strands waiting to complete, but there is no work
            wait            =  (self.inflight or (not node.is_deterministic) or self.visited or self.jump) and ((not keep_going_down) and  (not jump_back)) or node.squash

            # Condition 3: Nothing left to do
            sim_complete    = (not wait) and (not jump_back) and (not keep_going_down)

            # Create an in-flight entry
            if(not node.is_deterministic):
                self.inflight.append(threadID)

            # Release the resources of a thread: this is done when we are at a sink node's last visit
            release_thread  = is_sink_node or node.squash

            # Release resources the thread has occupied
            if(release_thread):
                self.release_strand(threadID)

            ###### ACTIONS #########

            # Next Address state machine
            # By default, we will assume we want to go down
            if(node.is_branch and node.branch_taken is not None):
                go_down_addr = node.branch_taken
            # Handles the Loop with 0 iterations case (uses goto)
            elif(node.goto != None and node.is_loop):
                go_down_addr = node.goto
            elif(node.num_children > 0):
                # Every visit takes you to a different children. In case of loops, there will be a single children (code size is compact)
                children_choice = 0 if(node.is_loop) else min(node.num_visits-node.num_parents, node.num_children-1)
                # Go down this path
                go_down_addr = node.cur_node.children[children_choice]
            else:
                go_down_addr = None

            # Defaults
            next_addr_fetch = go_down_addr
            next_addr_thread = go_down_addr

            # print("\n-------- Control Decision --------\n")
            if(keep_going_down):
                # If the node is a source node, we will create a new thread from here.
                # In case of while loops, there is a preface to execute the condition. And it should not be the last iteration.
                if((source_node_first_visit and (not was_visited)) or node.is_while):
                    threadID_next = self.get_threadID(threadID, source=True)
                    master_strand_next = threadID
                    self.threads_holding_resource.append(threadID_next)
                    # print("DEBUG: Created a new strand {0} for source node at master_strand {1}.".format(threadID_next, master_strand_next))
                # If sink node and last visit, we release the current thread and switch to the master thread
                elif(is_sink_node and is_last_visit):
                    threadID_next = master_strand
                    master_strand_next = self.levelup(master_strand)
                    # print("DEBUG Children strands complete, merging to {1}, {0}".format(master_strand, threadID))
                # else:
                    # print("DEBUG Will continue to retain the strand and keep going down")
            elif(jump_back):
                if(go_to_addr):
                        next_addr_fetch = node.goto
                        if(node.goto_level != None):
                            threadID_next = self.levelup(threadID, node.goto_level)
                            master_strand_next = self.levelup(threadID, node.goto_level+1)
                # Resume a previously started thread
                elif(resume_thread):
                    threadID_next, master_strand_next, next_addr_fetch = self.thread_stack.pop(0)
                    self.inflight.remove(threadID_next)
                    # print("DEBUG: Resuming an old strand with ID {0} at {1}".format(threadID, next_addr_fetch))
                # Create a new thread by jumping back to a source node.
                elif(create_thread):
                    jump_node  = self.jump[-1]
                    next_addr_fetch = int(jump_node.split("_")[0])
                    master_strand_next = '_'.join(jump_node.split("_")[1:-1])
                    is_new_thread = int(jump_node.split("_")[-1]) == 1
                    # Create a new strand
                    if(is_new_thread):
                        threadID_next = self.get_threadID(master_strand_next, source=True)
                        self.threads_holding_resource.append(threadID_next)
                    # Else recreate the strand
                    else:
                        threadID_next = master_strand_next
                        master_strand_next = self.levelup(master_strand_next)
                    # print("DEBUG: Forking off a new strand with ID {0} by jumping to {1}".format(threadID, next_addr_fetch))
                else:
                    self.active[issue_slot], next_addr_fetch, threadID_next, master_strand_next = False, None, None, None
                    # print("DEBUG Wait till someone completes")                    
            elif(wait):
                self.active[issue_slot], next_addr_fetch, threadID_next, master_strand_next = False, None, None, None
                # print("DEBUG Wait till someone completes")
            elif(sim_complete):
                print("\n\n\n \t\t SIMULATION FINISHED IN {0} CYCLES for PE{1}".format(self.env.now, self.name))
                self.active[issue_slot], next_addr_fetch, threadID_next, master_strand_next = False, None, None, None
                done = True
            else:
                sys.exit("Control Not Understood.")

            # print("\n\n INSTRUCTION : {0} at time {1} and thread {2} in PE {3}: slot {4}".format(node.instr, self.env.now, threadID, self.name, issue_slot))
            # print("Num Parents: {0}, Children: {1}, Num visits: {3}/{5} (Last visit: {4}) at master_strand {2}".format(node.num_parents, node.num_children, master_strand, node.num_visits, is_last_visit, node.num_visits_needed))
            # print("DEBUG keep going down: {0}, resume thread: {1}, create thread: {2}, wait: {3}, sim complete: {4}, release thread: {5}, Deterministic: {6} in PE{7}"\
            #         .format(keep_going_down, resume_thread, create_thread, wait, sim_complete, release_thread, node.is_deterministic , self.name))
            # print(" Sources: {2},  visited: {4}".format(self.inflight, self.thread_stack, self.jump, self.threads_holding_resource, self.visited))
            # print("Slot Status: {0}, {1}".format(self.active, self.addr_fetch))
            # print(self.resource_usage())
            # print("Inflight: {0}, Completed: {1}".format(self.inflight, self.thread_stack, self.jump, self.threads_holding_resource, self.visited))
            # input()

        # Slot is in a stalled state, until the stall is clear, we stay in the same insutrction
        elif(self.stalled[issue_slot]):
            threadID_next = threadID
            master_strand_next = master_strand
            next_addr_fetch = self.addr_fetch[issue_slot]
            done = False
            next_addr_thread = None

        # Slot is in wait state. If any ongoing threads finish, notify.
        else:
            # Defaults
            threadID_next = threadID
            master_strand_next = master_strand
            next_addr_fetch = None
            done = False
            next_addr_thread = None
            self.active[issue_slot] = False

            # If one of the previously launched thread is complete, we want to start there.
            if(self.thread_stack):
                threadID_next, master_strand_next, next_addr_fetch = self.thread_stack.pop(0)
                self.inflight.remove(threadID_next)
                self.active[issue_slot] = True
                # print("DEBUG: Resuming an old strand with ID: {0}, master ID: {1}, slot: {2}".format(threadID_next, master_strand_next, issue_slot))
            # Otherwise, if there is a place to jump to( that requires creating a new strand) and it is the master slot, it can go ahead.
            elif(master_slot and self.jump and self.are_resources_available()):
                jump_node  = self.jump[-1]
                next_addr_fetch = int(jump_node.split("_")[0])
                master_strand_next = '_'.join(jump_node.split("_")[1:-1])
                is_new_thread = int(jump_node.split("_")[-1]) == 1
                # Create a new strand
                if(is_new_thread):
                    threadID_next = self.get_threadID(master_strand_next, source=True)
                    self.threads_holding_resource.append(threadID_next)
                # Else recreate the strand
                else:
                    threadID_next = master_strand_next
                    master_strand_next = self.levelup(master_strand_next)
                self.active[issue_slot] = True
                # print("DEBUG: Forking off a new strand with ID {0} at {1}".format(threadID, next_addr_fetch))                
            # Otherwise, there is no option but to wait.
            elif(self.inflight or any(self.active) or self.jump or self.visited):
                pass
            # Check simulation end condition
            elif(not self.inflight):
                print("\n\n\n \t\t SIMULATION FINISHED IN {0} CYCLES for PE{1}".format(self.env.now, self.name))
                done = True

        # print("Active: {4}, Done: {0}, Next Address: {1}, Thread Resume Addr: {5}, Next threadID: {2}, next master_strand: {3}".format(done, next_addr_fetch, threadID_next, master_strand_next, self.active[issue_slot], next_addr_thread))

        return done, next_addr_fetch, threadID_next, next_addr_thread, master_strand_next

    def branch_miss(self, issue_slot=0, was_deterministic=False, sink_node=False):
        """
            This models the branch miss (not really a branch, but sort of), where the node fetched
            is different from the actual node.
        """

        # Check address that would be fetched (if it was wrong, we pay a penalty)
        # We pay a penalty of 2 cycles for mis-predictions on sink nodes/jumps, 1 cycle on loads/non-deterministic
        # instructions. This is because loads are know at decode stage itself.

        prev_addr = self.addr_fetch_prev[issue_slot]
        actual_addr = self.addr_fetch[issue_slot]
        new_prediction = None
        # If the slots were not active, we don't assess any penalty
        if((prev_addr == None) or (actual_addr == None) or self.stalled[issue_slot]):
            return 0

        # Fetch has two possible next addresses: simple increment/BTB prediction, higher priority to BTB (unless BTB miss)

        # Predicted Address at Fetch
        btb_hit = False# self.btb.hit(prev_addr)
        if(btb_hit):
            predicted = self.btb.predict(prev_addr)
            self.btb_logger.append([prev_addr, actual_addr, predicted])
        else:
            predicted = prev_addr + 1
            self.btb_logger.append([prev_addr, actual_addr, 'None'])

        # Correct Prediction, no penalty
        if(actual_addr == predicted):
            deficit = 0
            self.logger['btb']['hit'] += 1
            self.btb.update(prev_addr, actual_addr, True)
        # Pay the penalty and update BTB
        else:
            # At the decode stage, another attempt at prediction happens
            new_prediction = self.next_jump[issue_slot] if((not was_deterministic) or sink_node) else None
            deficit = 1 if(new_prediction == actual_addr) else 2
            # Update BTB
            self.btb.update(prev_addr, actual_addr, False)
            # self.btb[prev_addr] = actual_addr

        # print("\n Prev Addr: {4}, Address Next {0}, prediction {1}, New Prediction: {2}, deficit: {3}".format(actual_addr, predicted, new_prediction, deficit, prev_addr))
        # input()

        # Update the previous address
        self.addr_fetch_prev[issue_slot] = self.addr_fetch[issue_slot]
        self.logger['btb']['read'] += 1

        return deficit

    def read_operand(self, operand, threadID, increment=False):
        """ Return the operand value """
        operand_type, operand_value = operand.operand_type, operand.operand_value
        if((operand_type == 'register') or (operand_type == 'global register')):
            return self.regfile.read(operand_value)
        elif((operand_type == 'vector') or (operand_type == 'global vector')):
            data = self.vecfile.read(operand_value)
            assert(type(data) == list), "Read a {0}: {1} for a vector register".format(type(data), data)
            return data
        elif(operand_type == 'local register'):
            data = yield self.env.process(self.shared_regfile.read(operand, threadID, increment))
            return data
        elif(operand_type == 'local vector'):
            data = yield self.env.process(self.shared_vecfile.read(operand, threadID, increment))
            assert(type(data) == list), "Read a {0}: {1} for a vector register".format(type(data), data)
            # print("VEC data", data)
            return data
        elif(operand_type == 'loop counter'):
            data = yield self.env.process(self.shared_loopcounter.read(operand, threadID, increment))
            return data
        elif(operand_type == 'immediate'):
            return operand_value
        else:
            sys.exit("Operand type {0} unsupported".format(operand_type))

    def write_operand(self, operand, data, threadID):
        """ Return the operand value """
        operand_type, operand_value = operand.operand_type, operand.operand_value
        if((operand_type == 'register') or (operand_type == 'global register')):
            self.regfile.write(operand_value, data)
        elif((operand_type == 'vector') or (operand_type == 'global vector')):
            assert(type(data) == list), "Trying to write {0}: {1} for a vector register".format(type(data), data)
            self.vecfile.write(operand_value, data)
        elif(operand_type == 'local register'):
            yield self.env.process(self.shared_regfile.write(operand, data, threadID))
        elif(operand_type == 'local vector'):
            assert(type(data) == list), "Trying to write {0}: {1} for a vector register".format(type(data), data)
            yield self.env.process(self.shared_vecfile.write(operand, data, threadID))
        elif(operand_type == 'loop counter'):
            yield self.env.process(self.shared_loopcounter.write(operand, data, threadID))
        elif(operand_type == 'immediate'):
            operand_value
        else:
            sys.exit("Operand type {0} unsupported".format(operand_type))

    def get_level(self, threadID):
        """ 
            Returns the nesting level of the threadID
        """
        return len(threadID.split("_"))

    def levelup(self, threadID, level=1):
        """ 
            Returns the nesting level of the threadID
        """
        threadID = threadID.split("_")
        cur_level = len(threadID)
        trim = max(cur_level-level, 1)
        return '_'.join(threadID[:trim])

    def get_threadID(self, prefix, source=False):
        """ 
            Returns a unique strand ID.

            " Parameters
            source:  Whether a new strand is created or not. Example, go to a source node to create a new strand
            level: Requested level for the strand
            threadID: Current strand ID.
        """
        if(not source):
            prefix = prefix.split("_")[:-1]
        prefix = prefix + '_'

        # Get unique ID
        self.unique_id += 1
        return prefix + str(self.unique_id)

    def release_regreq(self, threadID):
        """
            Once a thread completes, we need to release all the registers it owned.
        """
        yield self.env.process(self.shared_regfile.release_all(threadID))
        yield self.env.process(self.shared_vecfile.release_all(threadID))
        yield self.env.process(self.shared_loopcounter.release_all(threadID))

    def release_strand(self, threadID):
        """
            Release a strand and its resources.
        """
        self.env.process(self.release_regreq(threadID))

        # Remove resources occupied by the thread
        if(threadID not in self.threads_holding_resource):
            print("WARNING: There may be ghost threads holding resource.")
            print("Was Releasing a strand with ID: {0}".format(threadID))
            sys.exit()
            # print("DEBUG: Releasing a strand with ID: {0}".format(threadID))
            # sys.exit("T:{0}, {1}".format(threadID, self.threads_holding_resource))
        else:
            self.threads_holding_resource.remove(threadID)

    def are_resources_available(self):
        """
            Before a new thread is created, minimum resources must be available.
        """
        regfile_available = self.shared_regfile.get_occupancy() < (self.parameters.SHARED_REGFILE_SIZE - self.parameters.MAX_REG_PER_THREAD)
        vecfile_available = self.shared_vecfile.get_occupancy() < (self.parameters.SHARED_VECFILE_SIZE - self.parameters.MAX_VEC_PER_THREAD)
        loopcounter_available = self.shared_loopcounter.get_occupancy() < (self.parameters.SHARED_LOOPCOUNTER_SIZE - self.parameters.MAX_LOOPS_PER_THREAD)
        available = regfile_available and vecfile_available and loopcounter_available

        return available

    def resource_usage(self):
        """
            Before a new thread is created, minimum resources must be available.
        """
        regfile = self.shared_regfile.get_occupancy() / (self.parameters.SHARED_REGFILE_SIZE)
        vecfile = self.shared_vecfile.get_occupancy() / (self.parameters.SHARED_VECFILE_SIZE)
        loopcounter = self.shared_loopcounter.get_occupancy() / (self.parameters.SHARED_LOOPCOUNTER_SIZE)
        thread_occupancy = len(self.inflight) + len(self.thread_stack) + 1
        lq_occupancy = self.active_loads # self.load_queue.occupancy()

        return regfile, vecfile, loopcounter, thread_occupancy, lq_occupancy

        # Check if enough registers are available

    def update_logger(self, payload):
        """
            Update the logger with the current values.
        """
        regfile_occupancy, vecfile_occupancy, loopc_occupancy, thread_occupancy, lq_occupancy = self.resource_usage()
        self.logger['regfile_occupancy'].append((self.env.now,regfile_occupancy))
        self.logger['thread_occupancy'].append((self.env.now,thread_occupancy))
        self.logger['lq_occupancy'].append((self.env.now,lq_occupancy))
        self.logger['ready_strands'].append((self.env.now, len(self.thread_stack)))

        # Find out the type of operation
        if(payload.op_type == 'reg'):
            self.logger['instr_type']['arith'] += 1
        elif((payload.op_type == 'load') or (payload.op_type == 'store') or (payload.op_type == 'storei')):
            self.logger['instr_type']['load'] += 1
        elif((payload.op_type == 'vload') or (payload.op_type == 'vstore')):
            self.logger['instr_type']['load'] += 16
        else:
            self.logger['instr_type']['control'] += 1

        # print("\n\n --- Status --- \n Strands in flight:{0}, strands that finished:{1} in PE {2}".format(len(self.inflight), len(self.thread_stack), self.name))
        # print("REGFILE Occupancy: {0}, vecfile occupancy: {1} Loop counter Occupancy {2} Thread Stack Occupancy {3} in PE {4}\n\n".format(\
        #      regfile_occupancy, vecfile_occupancy, loopc_occupancy, thread_occupancy, self.name))

    def get_datastructure_size(self):
        """ Find out the size of key data-structures."""
        visited_size        = len(self.visited)
        jump_size           = len(self.jump)
        inflight_size       = len(self.inflight)  # Store the threads in-flight
        thread_stack_size   = len(self.thread_stack) # In-flight threads store something once they complete

        sizes = {
                'visited_size' : visited_size,
                'jump_size' : jump_size, 
                'inflight_size' : inflight_size,
                'thread_stack_size' : thread_stack_size,}

        print("\n".join("{}\t{}".format(k, v) for k, v in sizes.items()))

    def get_memory_consumption(self):
        """
            Check the memory consumption by each object.
        """
        # Memory usage stats
        regfile_size        = sys.getsizeof(self.shared_regfile)
        vecfile_size        = sys.getsizeof(self.shared_vecfile)
        dcache_size         = sys.getsizeof(self.dcache)
        icache_size         = sys.getsizeof(self.icache)
        loopcounter_size    = sys.getsizeof(self.shared_loopcounter)
        logger_size         = sys.getsizeof(self.logger)
        visited_size        = sys.getsizeof(self.visited)
        jump_size           = sys.getsizeof(self.jump)
        inflight_size       = sys.getsizeof(self.inflight)  # Store the threads in-flight
        thread_stack_size   = sys.getsizeof(self.thread_stack) # In-flight threads store something once they complete
        resource_hold_size  = sys.getsizeof(self.threads_holding_resource)

        sizes = {
                'regfile_size': regfile_size,  
                'vecfile_size': vecfile_size,
                'dcache_size' : dcache_size,
                'icache_size' : icache_size,
                'loopcounter_size' : loopcounter_size,  
                'logger_size' : logger_size,
                'visited_size' : visited_size,
                'jump_size' : jump_size, 
                'inflight_size' : inflight_size,
                'thread_stack_size' : thread_stack_size,
                'resource_hold_size' : resource_hold_size,}

        print("\n".join("{}\t{}".format(k, v) for k, v in sizes.items()))
        input()

    def traversalEngine(self, BASE_ADDR):
        """
            This traverses the graph. Executes the fetch, decode, and dispatch stages of the pipeline.
        """
        self.INSTR_BASE_ADDR = BASE_ADDR# >> self.access_granularity_log)
        start_time = self.env.now

        # Create an independent process for store queue handling
        store_queue = self.env.process(self.store_launch())
        load_queue = self.env.process(self.load_launch())

        # Each step is a cycle of the pipeline.
        while(not all(self.done)):

            # One cycle
            yield self.env.process(self.step())

            # Add appropriate delay
            yield self.env.timeout(1)

        # Make sure the load-store queue process is complete too
        yield simpy.AllOf(self.env, [store_queue, load_queue,])

        # Log the time
        self.logger['total_cycles'] = self.env.now - start_time
        self.logger['end_sim'] = self.env.now
        # Utlization
        idle_time = float(self.logger['idle'])/self.logger['total_cycles']
        self.logger['engine_busy'] = 1 - idle_time
        self.logger['engine_idle'] = idle_time
