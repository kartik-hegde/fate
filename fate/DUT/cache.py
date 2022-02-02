import simpy
import random
import math
import copy
import sys
import os
import numpy as np
from copy import copy, deepcopy

class CacheModule():
    """ A realistic cache, models everything."""
    def __init__(self, env, next_level_cache, network_in, network_out, logger,
                outstanding_queue_size, HIT_LATENCY, MISS_LATENCY, ADDR_WIDTH,
                ACCESS_GRANULARITY, size=1024*32, linesize=64, associativity=math.inf,
                write_through=True, point_of_coherency=False,name='cache'):

        self.env = env
        self.next_level_cache = next_level_cache
        self.network_in = network_in
        self.network_out = network_out
        self.outstanding_queue_size = outstanding_queue_size
        self.outstanding_queue = simpy.Resource(self.env, outstanding_queue_size)
        self.HIT_LATENCY = HIT_LATENCY
        self.MISS_LATENCY = MISS_LATENCY
        self.associativity = associativity
        self.write_through = write_through
        self.name = name
        self.addr_width = ADDR_WIDTH
        self.access_granularity_bits = int(math.log(ACCESS_GRANULARITY,2))
        self.point_of_coherency = point_of_coherency
        self.logger = logger
        self.cur_time = self.env.now

        # Prepare the cache
        self.num_cache_lines = int(size/(linesize*associativity))
        self.linesize = linesize

        # Slicing the address for lookup
        self.bits_tag, self.bits_index, self.bits_offset = self.addr_slicing()
        self.index_mask = (2 ** self.bits_index)-1
        self.offset_mask = (2 ** self.bits_offset)-1

        # Cache data
        self.cacheline = [None for _ in range(2 ** (self.bits_offset-self.access_granularity_bits))]
        # Make sure the cache line is not currently busy
        self.cacheline_resource = [simpy.Resource(self.env,1) for _ in range(self.num_cache_lines)]

        self.data = [list(self.cacheline) for _ in range(self.num_cache_lines)] if(associativity==math.inf) else [[list(self.cacheline) for _ in range(associativity)] for _ in range(self.num_cache_lines)]
        self.tag = [None for _ in range(self.num_cache_lines)] if(associativity==math.inf) else [list([None for _ in range(associativity)]) for _ in range(self.num_cache_lines)]
        self.dirty = [0 for _ in range(self.num_cache_lines)] if(associativity==math.inf) else [list([0 for _ in range(associativity)]) for _ in range(self.num_cache_lines)]

        # Eviction Heuristic
        self.evict_queue = [deepcopy(EvictLine(self.num_cache_lines)),] if(associativity==math.inf) else [deepcopy(EvictLine(associativity))]*self.num_cache_lines

    def addr_slicing(self):
        """
            Define the tag look-up, index lookup and offset.
        """
        bits_index = int(math.ceil(math.log(self.num_cache_lines,2)))
        bits_offset = int(math.ceil(math.log(self.linesize,2)))
        bits_tag = self.addr_width - (bits_index + bits_offset)

        return bits_tag, bits_index, bits_offset

    def addr_construct(self, tag, index, offset):
        """ Generate a full address. """
        return (tag << (self.bits_index + self.bits_offset)) + (index << self.bits_offset) + (offset << self.access_granularity_bits)

    def get_sliced_addr(self, addr):
        """ Get the tag, index and offset portion. """
        ## <TAG> <INDEX> <OFFSET>

        # Right shift, no need of mask.
        tag = (addr >> (self.bits_index + self.bits_offset))
        # Right shift and apply mask.
        index = (addr>> (self.bits_offset)) & self.index_mask
        # Mask the upper bits, ignore the LSB corresponding to access granularity
        offset = (addr & self.offset_mask) >> self.access_granularity_bits
        # print(hex(addr), hex(tag), hex(index), hex(offset))
        return tag, index, offset

    def is_hit(self, tag, index):
        """ Check whether it is a hit in cache """
        # Check whether the tag is in the right index
        return tag in self.tag[index]

    def is_slot_available(self, index):
        """ Check if there is an empty slot."""
        return None in self.tag[index]

    def evict(self, index):
        """ Evict the address. if dirty, they only write it to lower hierarchy."""
        # This call returns the line to be evicted. (Invalidation should be handled by the caller)
        evict_idx = self.evict_queue[index].evict()
        # Check the dirty bit (set to 0)
        dirty = self.dirty[index][evict_idx]==1
        self.dirty[index][evict_idx] = 0

        return evict_idx, dirty

    def read(self, addr, cacheable=True, line_read=False):
        """ performs the cache read, either in this hierarchy/next one. """
        # Slice the address
        tag, index, offset = self.get_sliced_addr(addr)

        # Get the cache line (no one else is accessing)
        line_req = self.cacheline_resource[index].request()
        yield line_req

        # print("READ REQUEST at {0} for address {1}".format(self.name, addr))

        hit = self.is_hit(tag, index)

        # Hit.
        if(hit):
            # Slot where the hit is
            slot_idx = self.tag[index].index(tag)
            # Update eviction queue
            self.evict_queue[index].update(slot_idx)
            self.logger['read_hits'] += 1
            yield self.env.timeout(self.HIT_LATENCY)
            # Retrieve the data
            resp_data = self.data[index][slot_idx]

        # Else, next level of cache is called.
        else:
            self.logger['read_misses'] += 1
            # Make sure outstanding queue is available
            with self.outstanding_queue.request() as req:
                yield req
                # Log the occupancy
                if(self.env.now - self.cur_time > 100):
                    self.logger['queue_occupancy'].append((self.env.now, self.outstanding_queue.count/self.outstanding_queue_size))
                    self.cur_time = self.env.now
                # Cycles spent to check this cache
                yield self.env.timeout(self.MISS_LATENCY)
                # Access the next level cache and return the data
                resp_data = yield self.env.process(self.next_level_cache.read(addr, line_read=True))

            # Write to this cache only if cacheable.
            if(cacheable):
                # There is an empty slot already
                if(self.is_slot_available(index)):
                    slot_idx = self.tag[index].index(None)
                # Perfrom eviction
                else:
                    # Write this line back to lower levels, if dirty
                    slot_idx, dirty = self.evict(index)
                    if(dirty):
                        wr_addr = self.addr_construct(self.tag[index][slot_idx], index, 0)
                        self.env.process(self.next_level_cache_write(wr_addr, [*self.data[index][slot_idx]], line_write=True))
                    #     print("READ: Evicting a line from Cache {0} to Cache {1} with addr {3} data".format(self.name, self.next_level_cache.name, self.data[index][slot_idx], wr_addr))
                    # else:
                    #     wr_addr = self.addr_construct(self.tag[index][slot_idx], index, 0)
                    #     print("READ: Silent Evicting a line from Cache {0} at addr {2} data ".format(self.name, self.data[index][slot_idx], wr_addr))

                # Update the new data
                self.tag[index][slot_idx] = tag
                self.data[index][slot_idx] = resp_data
                # Update the eviction queue
                self.evict_queue[index].update(slot_idx)

        # Release the cache line
        self.cacheline_resource[index].release(line_req)
            
        # Make sure bandwidth is available
        yield self.env.process(self.upper_level_cache_resp(line_read))

        # Print the result
        # hit_or_miss = 'Hit' if(hit) else 'Miss'
        # print("Read {1} at Cache {0} for time: {2} address {3}, data: {4}".format(self.name, hit_or_miss, self.env.now, addr, resp_data[offset]))

        return [*resp_data] if(line_read) else resp_data[offset]

    def write(self, addr, data, bypass=False, line_write=False):
        """ Perform a write. Writes do not return any time delays"""
        # if(addr == 126615808):
        #     print("WRITE REQUEST at {0} for address {1} data {2}".format(self.name, addr, data))
        # Slice the address
        tag, index, offset = self.get_sliced_addr(addr)

        # Get the cache line (no one else is accessing)
        line_req = self.cacheline_resource[index].request()
        yield line_req

        # print("WRITE REQUEST at {0} for address {1} data {2}".format(self.name, addr, data))

        # Hit
        hit = self.is_hit(tag, index)

        # Print the result
        # hit_or_miss = 'Hit' if(hit) else 'Miss'
        # print("Write {1} at Cache {0} for time: {2} address {3}, writing {4}".format(self.name, hit_or_miss, self.env.now, addr, data))

        # Simply write to lower level
        if(bypass):
            yield self.env.process(self.next_level_cache_write(addr, data, line_write=line_write))
        else:
            # Hit.
            if(hit):

                # Hit latency
                yield self.env.timeout(self.HIT_LATENCY)
                self.logger['write_hits'] += 1

                slot_idx = self.tag[index].index(tag)
                if(line_write):
                    self.data[index][slot_idx] = [*data]
                else:
                    self.data[index][slot_idx][offset] = data

            # Miss
            else:
                # Cycles spent to check this cache
                yield self.env.timeout(self.MISS_LATENCY)
                self.logger['write_misses'] += 1

                # Get a slot, if available.
                if(self.is_slot_available(index)):
                    slot_idx = self.tag[index].index(None)

                # Eviction needed
                else:
                    slot_idx, dirty = self.evict(index)
                    # Slot had dirty data, need to push it down
                    if(dirty):
                        wr_addr = self.addr_construct(self.tag[index][slot_idx], index, 0)
                        self.env.process(self.next_level_cache_write(wr_addr, [*self.data[index][slot_idx]],line_write=True))
                    #     print("WRITE: Evicting a line from Cache {0} to Cache {1} with addr {3} data".format(self.name, self.next_level_cache.name, self.data[index][slot_idx], wr_addr))
                    # else:
                    #     wr_addr = self.addr_construct(self.tag[index][slot_idx], index, 0)
                    #     print("WRITE: Silent Evicting a line from Cache {0} at addr {2} data ".format(self.name, self.data[index][slot_idx], wr_addr))

                # If we are writing an entire line, then things are simpler. Evict/find a free slot, and insert the line.
                if(line_write):
                    new_line = [*data]
                # If we are not writing a line and it is a miss, we will need to bring the updated line from the lower levels.
                else:
                    # if(addr > 100000000):
                    #     print("MYSTERIOUS WRITE")
                    # print("READING new line ", addr)
                    new_line = yield self.env.process(self.next_level_cache.read(addr, line_read=True))
                    # Update with new data
                    new_line[offset] = data

                # Update the new data
                self.tag[index][slot_idx] = tag
                self.data[index][slot_idx] = new_line

            # Update the dirty flag
            self.dirty[index][slot_idx] = 1
            # Update the eviction queue
            self.evict_queue[index].update(slot_idx)

            # Release the cache line
            self.cacheline_resource[index].release(line_req)

            # Write through cache
            if(self.write_through):
                yield self.env.process(self.next_level_cache_write(addr, data, line_write=line_write))


            # Success
            return True

    def deep_read(self, addr, cacheable=True, line_read=False):
        """ Performs a deep read, all the way to the point of coherency."""
        # If we are at the PC, then simply perform a read.
        if(self.point_of_coherency):
            resp = yield self.env.process(self.read(addr, cacheable, line_read))
        # Else launch a read from a lower level (irrespective of whether it is a hit or a miss at the local cache)
        else:
            resp = yield self.env.process(self.next_level_cache.deep_read(addr, cacheable, line_read=line_read))

        return resp

    def deep_write(self, addr, data, cacheable=True, line_write=False):
        """ Performs a deep read, all the way to the point of coherency."""
        # If we are at the PC, then simply perform a read.
        if(self.point_of_coherency):
            resp = yield self.env.process(self.write(addr, data, line_write=line_write))
        # Else launch a read from a lower level (irrespective of whether it is a hit or a miss at the local cache)
        else:
            resp = yield self.env.process(self.next_level_cache.deep_write(addr, data, line_write=line_write))

        return resp

    def upper_level_cache_resp(self, line_read=False):
        """ This function makes sure the Network bandwidth is satisfied."""
        # Is bandwidth available
        yield self.env.process(self.network_in.transfer(self.linesize if(line_read) else 2 ** self.access_granularity_bits, write=False))

    def next_level_cache_write(self, addr, data, line_write=False):
        """ This function makes sure the Network bandwidth is satisfied."""
        # print("EVICITNG in {0} for address {1}".format(self.name, addr))
        # Is bandwidth available
        yield self.env.process(self.network_out.transfer(self.linesize if(line_write) else 2 ** self.access_granularity_bits, write=True))
        # Return data
        yield self.env.process(self.next_level_cache.write(addr, data, line_write=line_write))

    def invalidate_cache(self):
        """ Invalidate the cache"""
        self.tag = [None for _ in range(self.num_cache_lines)] if(self.associativity==math.inf) else [[None for _ in range(self.associativity)] for _ in range(self.num_cache_lines)]
        self.dirty = [0 for _ in range(self.num_cache_lines)] if(self.associativity==math.inf) else [[0 for _ in range(self.associativity)] for _ in range(self.num_cache_lines)]
        self.evict_queue = [EvictLine(self.num_cache_lines),] if(self.associativity==math.inf) else [EvictLine(self.associativity)]*self.num_cache_lines

    def preload_data(self,data, base_addr):
        """ Preload to a cache """
        for i,d in enumerate(data):
            yield self.env.process(self.write(base_addr+4*i,d))

    def preload_cache(self, data, tag):
        """
            Preload the cache with some data.
        """
        assert type(data)==list, "Input to preload should be a list not a {0}".format(type(data))
        assert type(tag)==list, "Input to preload should be a list not a {0}".format(type(tag))

        # Make sure the size and associativity match.
        if(self.associativity ==  math.inf):
            assert (len(data)==self.num_cache_lines), "Data does not match cache size"
            assert (len(tag)==self.num_cache_lines), "Tag does not match cache size"
        else:
            associativity, lines = len(data[0]), len(data)
            assert ((associativity==self.associativity) and (lines==self.num_cache_lines)), "Associativity {0} or size {1} does not match cache size".format(associativity, lines)

        self.data = data
        self.tag = tag

    def cache_dump(self, path=os.getcwd()):
        """ Dump the current cache state. """
        np.save(path + 'cacheDump_' + self.name + '.npy', [self.data, self.tag, self.evict])




class SimpleCacheModule():
    """ A simple statistical cache. Returns dummy data, uses probability to decide hit/miss."""

    def __init__(self, env, hitrate, outstanding_queue_size, HIT_LATENCY, MISS_LATENCY, size=1024*32):
        self.env = env
        self.hitrate = hitrate
        self.outstanding_queue = simpy.Resource(self.env, outstanding_queue_size)
        self.HIT_LATENCY = HIT_LATENCY
        self.MISS_LATENCY = MISS_LATENCY
        self.data = [0]*size

    def preload(self, data):
        """
            Preload the cache with some data.
        """
        assert type(data)==list, "Input to preload should be a list not a {0}".format(type(data))
        self.data = data

    def access(self, addr):
        """
            All requestors should call this to ensure the delay.
            Then call read/write appropriately.
        """
        roll_dice = random.random() < self.hitrate
        # Hit
        if(roll_dice):
            yield self.env.timeout(self.HIT_LATENCY)
        else:
            yield self.env.timeout(self.MISS_LATENCY)

    def read(self, addr):
        """
            Read the actual data.
        """
        return self.data[addr]

    def write(self, addr, data):
        """
            Silent write.
        """
        self.data[addr] = data
        return None

class EvictLine:
    """ Stores the next line to evict. """
    def __init__(self, num_blocks, strategy='LRU'):
        self.num_blocks = num_blocks
        # Represents the ranking of each entry.
        # if order[0] = 1, 1st line was most recently accessed.
        self.order = list(range(num_blocks))
        self.strategy = strategy

    def update(self, idx):
        """ Update the data structure based on what was recently accessed."""
        # Current ranking of the index
        # rank = self.order.index(idx)

        # Remove and insert the idx at the head
        self.order.remove(idx)
        self.order.insert(0,idx)

    def evict(self):
        """ Find the position to evict and send it to head of the order."""
        evict_idx = self.order[-1]
        # print("Attempting evict a line whose evict queue is {0}, hence line choice is {1}".format(self.order, evict_idx))

        # Update the order
        self.update(evict_idx)
        # print("Updated Order is {0}".format(self.order))

        return evict_idx

class Test:

    def __init__(self):
        self.cache_size = 32*1024
        self.linesize = 64
        self.associativity = 4
        self.num_cache_lines = (self.cache_size)/(self.linesize * self.associativity)
        self.addr_width = 32
        self.access_granularity_bits = int(math.log(4,2))
        self.bits_tag, self.bits_index, self.bits_offset = self.addr_slicing()
        self.index_mask = (2 ** self.bits_index)-1
        self.offset_mask = (2 ** self.bits_offset)-1

    def addr_slicing(self):
        """
            Define the tag look-up, index lookup and offset.
        """
        bits_index = int(math.ceil(math.log(self.num_cache_lines,2)))
        bits_offset = int(math.ceil(math.log(self.linesize,2)))
        bits_tag = self.addr_width - (bits_index + bits_offset)
        print(bits_tag, bits_index, bits_offset)
        return bits_tag, bits_index, bits_offset

    def addr_construct(self, tag, index, offset):
        """ Generate a full address. """
        print(hex(tag << (self.bits_index + self.bits_offset)), hex(index << self.bits_offset), hex(offset << self.access_granularity_bits))
        return (tag << (self.bits_index + self.bits_offset)) + (index << self.bits_offset) + (offset << self.access_granularity_bits)

    def get_sliced_addr(self, addr):
        """ Get the tag, index and offset portion. """
        ## <TAG> <INDEX> <OFFSET>

        # Right shift, no need of mask.
        tag = (addr >> (self.bits_index + self.bits_offset))
        # Right shift and apply mask.
        index = (addr>> self.bits_offset) & self.index_mask
        # Mask the upper bits, ignore the LSB corresponding to access granularity
        offset = (addr & self.offset_mask) >> self.access_granularity_bits
        print(hex(addr), hex(tag), hex(index), hex(offset))
        return tag, index, offset

#### UNIT TEST ####
if __name__ == '__main__':

    from network import Network
    # Simple Cache test
    # def read(env, name, cache):
    #     with cache.outstanding_queue.request() as req:
    #         print("Access starting for Read {0} at {1}".format(name, env.now))
    #         yield req

    #         yield env.process(cache.access(0))
    #         print("Access completed for Read {0} at {1}".format(name, env.now))

    # def create_req(env, cache, number, interval):
    #     for i in range(number):
    #         yield env.timeout(interval)
    #         env.process(read(env, str(i), cache))

    # env = simpy.Environment()
    # cache = Cache(env, 0.8, 5)
    # env.process(create_req(env, cache, 10, 5))
    # env.run(until=100)

    # Complete cache test
    env = simpy.Environment()

    # NoC PE <--> L1
    noc_pe_l1 = Network(env,10000000,0) # Instant, infinite bandwidth
    # NoC L1 <--> L2
    noc_l1_l2 = Network(env,1000,15)
    # NoC L2 <--> DRAM
    noc_l2_dram = Network(env,200,100)

    # 64 MB
    dram = DRAM(env,noc_l2_dram, size=1024*1024*64,linesize=64, addr_width=32, access_granularity=4, LATENCY=200)
    data = list(range(1024*1024))
    data = [data[16*i:16*(i+1)] for i in range(len(data)//16)]
    dram.preload(data, 0)

    # L2
    l2_cache = CacheModule(env, next_level_cache=dram, network_in=noc_l1_l2, network_out=noc_l2_dram, outstanding_queue_size=16, HIT_LATENCY=30, MISS_LATENCY=2, ADDR_WIDTH=32,
                 ACCESS_GRANULARITY=4, size=1024*1024, linesize=64, associativity=8, name='L2')
    # l1 Cache
    l1_cache = CacheModule(env, next_level_cache = l2_cache, network_in=noc_pe_l1, network_out=noc_l1_l2, outstanding_queue_size=4, HIT_LATENCY=1, MISS_LATENCY=1, ADDR_WIDTH=32,
                 ACCESS_GRANULARITY=4, size=1024*8, linesize=64, associativity=4, name='L1')

    # Read Test
    def read_test():

        ##### CHECK IF A LINE IS READ CORRECTLY
        # Read a line, check spatial reuse
        addresses = 0x0, 0x4, 0x8

        # Read the first address, which should be a miss
        tag, index, offset = l1_cache.get_sliced_addr(addresses[0])
        assert l1_cache.is_hit(tag, index) == False, "TEST FAILED, READ MISS EXPECTED"
        d = yield env.process(l1_cache.read(addresses[0]))

        # The second address should hit, since they map to the same line
        tag, index, offset = l1_cache.get_sliced_addr(addresses[1])
        assert l1_cache.is_hit(tag, index) == True, "TEST FAILED, READ HIT EXPECTED"

        print("READ LINE TEST PASSED")

        ##### CHECK IF INDEX WORKS AS EXPECTED
        # Read 10 addresses with different index
        data = []
        addresses = 0x40, 0x50, 0x60, 0x70
        for i in addresses:
            addr = i
            # All misses
            d = yield env.process(l1_cache.read(addr))
            data.append(d)

        # Read the same 10 addresses, all hit
        check_data = []
        for i in addresses:
            addr = i
            tag, index, offset = l1_cache.get_sliced_addr(addr)
            assert l1_cache.is_hit(tag, index) == True, "TEST FAILED, READ HIT EXPECTED"
            d = yield env.process(l1_cache.read(addr))
            check_data.append(d)
        assert data == check_data, "TEST FAILED, READ MISS"

        print("READ LINE INDEX PASSED")

        # Read miss and evict test. L1 cache has associativity of 4.
        # Miss 5 addresses of the same index, now the first address must be a miss.
        addresses = 0x800, 0xC00, 0x3000, 0x1000, 0x2000
        for addr in addresses:
            yield env.process(l1_cache.read(addr))
        addr = addresses[0]
        # Should be missed in L1
        tag, index, offset = l1_cache.get_sliced_addr(addr)
        assert l1_cache.is_hit(tag, index) == False, "TEST FAILED, READ MISS EXPECTED"
        # Should hit in L2
        tag, index, offset = l2_cache.get_sliced_addr(addr)
        assert l2_cache.is_hit(tag, index) == True, "TEST FAILED, READ MISS EXPECTED"

        print(" Read Test Completed at: ", env.now)
        l1_cache.invalidate_cache()

        print("\n\n \t\t  READ TEST PASSED \n\n")

    # Write test
    def write_test():

        ### 1. Test WRITE MISS FILL (Both, same line and across the line)
        # Write to 10 addresses
        data = list(range(6))
        addresses =  0x4, 0x8, 0xC, 0x24, 0x28, 0x2C
        for idx,addr in enumerate(addresses):
            addr = addr
            # All writes miss
            yield env.process(l1_cache.write(addr, data[idx], bypass=False))

        # Read all of them (they all should hit)
        check_data = []
        for idx,addr in enumerate(addresses):
            addr = addr
            tag, index, offset = l1_cache.get_sliced_addr(addr)
            assert l1_cache.is_hit(tag, index) == True, "TEST FAILED, READ HIT EXPECTED"
            d = yield env.process(l1_cache.read(addr))
            check_data.append(d)
        assert data == check_data, "TEST FAILED, DATA DID NOT MATCH"

        ### 2. Test WRITE MISS EVICT
        addresses = 0x800, 0xC00, 0x3000, 0x1000, 0x2000
        for idx,addr in enumerate(addresses):
            yield env.process(l1_cache.write(addr, data[idx], bypass=False))
        addr = addresses[0]
        # Should be missed in L1
        tag, index, offset = l1_cache.get_sliced_addr(addr)
        assert l1_cache.is_hit(tag, index) == False, "TEST FAILED, READ MISS EXPECTED"
        # Should hit in L2
        tag, index, offset = l2_cache.get_sliced_addr(addr)
        assert l2_cache.is_hit(tag, index) == True, "TEST FAILED, READ MISS EXPECTED"

        ### 3. WRITE HIT
        # Write to an address
        yield env.process(l1_cache.write(0x804, 0xDEADBEEF, bypass=False))
        # Write again, this must hit.
        yield env.process(l1_cache.write(0x804, 0xDEADBABE, bypass=False))
        # Read should return the new value and hit
        tag, index, offset = l1_cache.get_sliced_addr(0x804)
        assert l1_cache.is_hit(tag, index) == True, "TEST FAILED, READ HIT EXPECTED"
        d = yield env.process(l1_cache.read(0x804))
        assert d==0xDEADBABE, "TEST FAILED, UNEXPECTED DATA"

        print("\n\n \t\t  WRITE TEST PASSED \n\n")
    # Read test
    env.run(env.process(read_test()))
    env.run(env.process(write_test()))
    print("\n\n \t\t  TESTS PASSED \n\n")
