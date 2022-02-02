import simpy
import random
import math
import copy
import sys
import os
import numpy as np
from copy import copy, deepcopy

class DRAM:
    """ Represents DRAM. Assumption is everything fits here."""
    def __init__(self, env, network_in, size, linesize, addr_width, access_granularity, LATENCY):
        self.env = env
        self.name = 'DRAM'
        self.network_in = network_in
        self.linesize = linesize
        self.latency = LATENCY
        self.addr_width = addr_width
        num_lines = size//linesize
        self.access_granularity_bits = int(math.log(access_granularity,2))
        self.cacheline = [1 for _ in range(self.linesize//access_granularity)]
        self.data = [list(self.cacheline) for i in range(num_lines)]
        self.DUMMY_DATA = [0x0 for _ in range(self.linesize//access_granularity)]
        self.bits_offset = int(math.ceil(math.log(self.linesize,2)))
        self.offset_mask = (2 ** self.bits_offset)-1
        self.open_lines = {}

    def get_sliced_addr(self, addr):
        """ Get the Addr and offset portion. """
        ## <LINE> <OFFSET>
        # Right shift, no need of mask.
        line = addr >> self.bits_offset
        # Mask the upper bits, ignore the LSB corresponding to access granularity
        offset = (addr & self.offset_mask) >> self.access_granularity_bits
        return line, offset

    def preload(self, data, base=0):
        assert (type(data) == list), "A list should be preloaded"
        print("Preloading DRAM with {0} Bytes of data at {1}".format(len(data)*self.linesize, base))
        for addr in range(len(data)):
            yield self.env.process(self.write(base+(addr<<self.bits_offset), [*data[addr]], line_write=True))
        # print("DRAM Preload Complete")

    def read(self, addr, line_read=True):
        # Slice the address
        line, offset = self.get_sliced_addr(addr)

        # Make sure the DRAM is preloaded with the requested data
        # assert (len(self.data) >= addr), "Reading DATA from DRAM beyond preloaded: {0}".format(addr)
        # print("READ {1} at {0} for time: {2} address {3}, reading {4}".format(self.name, len(self.data) >= line, self.env.now, addr, self.data[line][offset]))

        # If the line is already open and being written, read right away
        while(line in self.open_lines):
            yield self.env.timeout(1)

        # Latency to access the DRAM
        yield self.env.timeout(self.latency)

        # Make sure we have the bandwidth
        yield self.env.process(self.upper_level_cache_resp())

        # assert (line <len(self.data)), "Accessing data beyond valid region: {0}".format(addr)

        if(line_read):
            return [*self.data[line]] if(line <len(self.data)) else [*self.DUMMY_DATA]
        else:
            return self.data[line][offset] if(line<len(self.data)) else 0x1


    def write(self, addr, data, line_write=False):
        """ Write to DRAM"""

        # Slice the address
        line, offset = self.get_sliced_addr(addr)

        # if(addr == 17190948):
        # print("Write {1} at {0} for time: {2} address {3}, writing {4}".format(self.name, len(self.data) >= line, self.env.now, addr, data))

        # Make sure write is withing range.
        self.open_lines[line] = data

        # assert (len(self.data) >= line), "Writing DATA to DRAM beyond preloaded"
        yield self.env.timeout(self.latency)

        # Write
        try:
            if(line_write):
                self.data[line] = [*data]
            else:
                self.data[line][offset] = data
        # When writing beyond the memory size. We will ignore it.
        except:
            # print("WARNING! Address {0} is beyond the DRAM simulated size.".format(addr))
            # print("Write {1} at {0} for time: {2} address {3}, writing".format(self.name, len(self.data) >= line, self.env.now, addr, data))
            # sys.exit()
            pass

        del self.open_lines[line]

        # print(addr, line, offset, len(self.data[line]))
        # input()
        return True

    def upper_level_cache_resp(self):
        """ This function makes sure the Network bandwidth is satisfied."""
        # Is bandwidth available
        yield self.env.process(self.network_in.transfer(self.linesize))