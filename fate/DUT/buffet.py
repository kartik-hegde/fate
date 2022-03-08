import simpy

from fate.parameters import Parameters

class Buffet:
    """Sequential Fill - Random Access - Random Update"""

    def __init__(self, env, parameters, name, size, network) -> None:
        self.env = env
        self.params = parameters
        self.name = name
        self.size = size
        self.network = network
        self.head = 0
        self.tail = 0
        self.credits = simpy.Container(env, capacity=size, init=size)
        self.valid_data = simpy.Container(env, capacity=size, init=0)
        self.buffet = [0 for _ in range(size)]
        
    def read(self, addr, shrink=False):
        """
            Read data from current head + offset. 
            Any read that is out of valid space will block.
        """
        # Time penalty
        yield self.env.timeout(self.params.BUFFET_R_LATENCY)
        # Make sure the data exists
        yield self.valid_data.get(1)
        # Check if it is within valid region
        assert self._within_valid_region(addr), "Invalid address"
        # Calculte the actual address
        addr = (self.head + addr)%self.size
        # Shrink if requested
        if(shrink):
            yield self.env.process(self.shrink(1))
        # Read from Buffet
        value = self.buffet[addr]
        # Return the value
        # print("Read {0} from Addr {1} at {2}. Credits: {3}".format(value, addr, self.name, self.credits.level))
        # Network latency
        yield self.env.process(self.network_log(write=False))
        return value

    def fill(self, value, latency=0):
        """
            Fill is done sequentially. A block of data is also accepted.

            If tail exceeds head, then valid data is overwritten.

            Assumption: sender has checked credits to ensure that there is space.
        """
        # Network latency
        yield self.env.process(self.network_log(write=True))
        # Time penalty (Aports idea to be able to add latency)
        yield self.env.timeout(latency + self.params.BUFFET_W_LATENCY)
        # Reduce credits 
        yield self.credits.get(1)
        # Increase valid region
        yield self.valid_data.put(1)

        # Write and increment tail
        # print("Writing to {0} with {1}".format(self.tail, value))
        self.buffet[self.tail] = value
        # print("Wrote {0} to Addr {1} at {2}. Credits: {3}".format(value, self.tail, self.name, self.credits.level))
        self.tail = (self.tail + 1) % self.size
        # self.credits -= 1

    def shrink(self, size):
        """
            Move head ahead. Shrink is capped to the current valid size.
        """
        # Check if you are shrinking too much
        assert size <= self._valid_region_size(), "Trying to shrink more than valid region ({0} vs {1}".format(self._valid_region_size(), size)
        # Head wraps around
        self.head = (self.head + size) % self.size
        # Add credits
        yield self.credits.put(size)
        # print("Shrunk {0} at {1}. Credits: {2}".format(size, self.name, self.credits.level))

    def update(self, addr, value, latency=0):
        """
            Update performed within valid space. If it exceeds valid space, request ignored.    
        """
        # Time penalty (Aports idea to be able to add latency)
        yield self.env.timeout(latency + self.params.BUFFET_W_LATENCY)

        assert self._within_valid_region(addr), "Invalid address"
        # Calculte the actual address
        addr = (self.head + addr)%self.size
        # Update value
        # print("Updating {0} with {1}".format(addr, value))
        self.buffet[addr] = value

    def bulk_fill(self, values):
        "Fills bulk data using the fill routine"
        # Convert to list
        values = values if(type(values) == list) else [values]
        # Fill values
        for _, val in enumerate(values):
            yield self.env.process(self.fill(val))

    def bulk_read(self, addresses):
        """Reads in bulk"""
        values = []
        # Convert to list
        addresses = addresses if(type(addresses) == list) else [addresses]
        # read values
        for addr in addresses:
            val = yield self.env.process(self.read(addr))
            values.append(val[0])
        return values

    def get_credits(self):
        """Returns total credits"""
        return self.credits.level

    def is_empty(self):
        """Check if empty"""
        return self.head == self.tail

    def _within_valid_region(self, addr):
        """ Check if addr is within valid region"""
        return addr <= (self.size - self.credits.level)

    def _valid_region_size(self):
        """Return the size of the valid region""" 
        return self.size - self.credits.level

    def _is_full(self):
        """Return if full"""
        return self.head == ((self.tail + 1)%self.size)

    def network_log(self, write=False):
        """ This function makes sure the Network bandwidth is satisfied."""
        # Is bandwidth available
        yield self.env.process(self.network.transfer(self.params.CACHELINE_SIZE, write=write))


if __name__ == "__main__":
    import random
    print("Running Buffet Tests")

    def basic_routines(env, buffet):
        # Read test
        read_data = yield env.process(buffet.bulk_read(list(range(0,5)))) 
        assert  read_data == list(range(0,5)), "Read Test Failed. Got {0} instead of {1}".format(read_data, list(range(0,5)))
        print("\n\nBasic read test passed.")

        # Update Test
        new_data = [val+1 for val in range(0,5)]
        for addr in range(0,5):
            yield env.process(buffet.update(addr, new_data[addr]))
        read_data = yield env.process(buffet.bulk_read(list(range(0,5)))) 
        assert read_data == new_data, "Update Test Failed"
        print("\n\nUpdate test passed.")

        # Shrink test
        yield env.process(buffet.shrink(2))
        expected_data = new_data[2:]
        read_data = yield env.process(buffet.bulk_read(list(range(0,3)))) 
        assert  read_data == expected_data, "Shrink Test Failed. Expected {0}, but got {1}".format(expected_data, read_data)
        print("\nShrink test passed.")

    def delayed_update(env, buffet, addr, lock):
        yield env.timeout(100)
        yield env.process(buffet.update(addr,0xDEADBEEF,release=lock))

    def update_test(env, buffet):
        # Read with update flag
        _, lock = yield env.process(buffet.read(1, will_update=True))
        # Delayed update process
        env.process(delayed_update(env, buffet, 1, lock))
        # Create another read process. It should block and not finish until after update is done.
        new_data = yield env.process(buffet.read(1))
        assert new_data[0] == 0xDEADBEEF, "Failed Update lock test. Expected {0}, got {1}".format(0xDEADBEEF, new_data)
        print("Update Lock Test Passed")

    env = simpy.Environment()
    parameters = Parameters()

    # -- Basic test --
    print("\n\n-------- Basic Tests ----------- \n\n")
    # Instantiate and fill random data
    buffet = Buffet(env, parameters, 10)
    proc = env.process(buffet.bulk_fill(list(range(0,5))))
    env.run(proc)
    proc = env.process(basic_routines(env, buffet))
    env.run(proc)

    # -- Wrap Around test --
    print("\n\n-------- Wrap Around Tests ----------- \n\n")
    # Instantiate and fill buffet with 8 elements
    buffet = Buffet(env, parameters, 10)
    proc = env.process(buffet.bulk_fill(list(range(0,8))))
    env.run(proc)
    # Shrink by 8
    proc = env.process(buffet.shrink(8))
    env.run(proc)
    # Fill and test
    proc =  env.process(buffet.bulk_fill(list(range(0,5))))
    env.run(proc)
    proc = env.process(basic_routines(env, buffet))
    env.run(proc)

    # -- Update Lock test --
    print("\n\n-------- Update Lock Tests ----------- \n\n")
    proc = env.process(update_test(env, buffet))
    env.run(proc)

