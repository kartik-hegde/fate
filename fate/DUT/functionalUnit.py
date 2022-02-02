"""
    Define functional Units.
"""
import simpy
import math
import sys

class SLSFunctionalUnit(FunctionalUnit):
    """ SLS Operation. Load and Add. """

    def __init__(self, env, dcache, regfile, vecfile, shared_regfile, shared_vecfile, count, latency, cacheline_size, simd_width):
        super().__init__(env, dcache, regfile, vecfile, shared_regfile, shared_vecfile, count, latency)
        self.cacheline_size = cacheline_size
        self.simd_width = simd_width

    def run(self, operands, threadID=0):
        """ Runs the operations. """

        total_cycles = 0

        # Read Operands
        ACC_REG = yield self.env.process(self.read_operand(operands[0], threadID))
        ET_ADDR = yield self.env.process(self.read_operand(operands[1], threadID))
        ET_WIDTH = yield self.env.process(self.read_operand(operands[2], threadID))

        # Number of loads needed to get the entire ET line
        num_load_iters = int(math.ceil(ET_WIDTH/self.cacheline_size))
        # Number of SIMD operations needed
        num_compute_iters = int(math.ceil(ET_WIDTH/self.simd_width))
        # print("SLSFU number of iterations {0}, {1}".format(num_compute_iters, num_load_iters))
        for _ in range(num_compute_iters):
            for l in range(num_load_iters):
                # print("SLSFU Load {0} addr {1} start thread {2}".format(l, ET_ADDR, threadID))
                # Access the cache (load)
                read_data = yield self.env.process(self.dcache.read(ET_ADDR))
                # Cycles spent to issue a load
                total_cycles += 1
                # Increment the address
                ET_ADDR += 4
                print("SLSFU Load {0} done thread {1}".format(l,threadID))

            # Accumulate (Vector Sum)
            result = read_data
            # Cycles spent to issue a load
            total_cycles += 1
            # Perform the sum
            yield self.env.timeout(self.latency)

        # Write to the Vec reg file
        yield self.env.process(self.write_operand(operands[0], result, threadID))
        # print("SLSFU Completed for thread {0}".format(threadID))

        return total_cycles
