"""
    This file defines a simple bandwidth constrained network on-chip.
"""

import simpy

class Network:

    def __init__(self, env, BYTES_PER_CYCLE, LATENCY, logger):
        self.env = env
        # At any point, the network can have bytes/cycle (bandwidth) * Latency number of data in it.
        self.max_inflight = BYTES_PER_CYCLE * LATENCY
        self.capacity = simpy.Resource(self.env, self.max_inflight)
        self.latency = LATENCY
        self.logger = logger
        self.cur_time = self.env.now

    def transfer(self, BYTES, write=False):
        """ Transfers the requested number of bytes."""

        # Grab the resource
        with self.capacity.request() as req:
            yield req
            self.update_logger(BYTES, write)
            yield self.env.timeout(self.latency)

    def update_logger(self, BYTES, write=False):
        # Update the loggers
        if(write):
            self.logger['write_bytes_transferred'] += BYTES
        else:
            self.logger['read_bytes_transferred'] += BYTES

        if(self.env.now - self.cur_time > 100):
            self.logger['utilization'].append((self.env.now, float(self.capacity.count)/self.max_inflight))
            self.cur_time = self.env.now
