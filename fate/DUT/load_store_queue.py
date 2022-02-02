import simpy

class StoreQueue:
    """
        Store queue. Memory consistency.
    """
    def __init__(self, env, queue_size=32):
        self.store_queue = []
        self.stores_inflight = {}
        self.queue_size = queue_size
        self.slots = simpy.Resource(env, capacity=queue_size)

    def enqueue(self, args):
        """
            Add an entry.
        """
        addr = args[0]
        data = args[1]
        metadata = args[2:]
        # Check if a store is already in flight
        if(addr in self.stores_inflight):
            # Update the existing entry
            self.stores_inflight[addr][0] = data
            
        else:
            # Add a new entry
            req = self.slots.request()
            yield req
            self.store_queue.append([addr, metadata])
            self.stores_inflight[addr] = [data, req]

    def dequeue(self):
        """
            Dequeue an entry (if not empty).
        """
        if(self.store_queue):
            # Get the first entry
            addr, metadata = self.store_queue.pop(0)
            # Remove the in-flight
            data, req = self.stores_inflight.pop(addr, None)
            # Release the slot
            self.slots.release(req)

            return (addr, data, *metadata)
        else:
            return None

    def check(self, addr):
        """
            Check if there is an inflight store for given addr.
        """
        if(addr in self.stores_inflight):
            return self.stores_inflight[addr][0]
        else:
            return None

    def occupancy(self):
        """
            Check queue occupancy (returns a ratio)
        """
        return self.slots.count/self.queue_size

    def empty(self):
        """
            Check if the queue is empty
        """
        return len(self.store_queue)==0


class LoadQueue:
    """
        Load queue. Memory consistency.
        We are not using Python inbuilt queue to support Simpy .
    """
    def __init__(self, env, queue_size=32):
        self.load_queue = []
        self.queue_size = queue_size
        self.slots = simpy.Resource(env, capacity=queue_size)

    def enqueue(self, packet):
        """
            Add an entry.
        """
        # Add a new entry
        req = self.slots.request()
        yield req
        self.load_queue.append([packet, req])

    def dequeue(self):
        """
            Dequeue an entry (if not empty).
        """
        if(self.load_queue):
            # Get the first entry
            packet, req = self.load_queue.pop(0)
            # Release the slot
            self.slots.release(req)

            return packet
        else:
            return None

    def occupancy(self):
        """
            Check queue occupancy (returns a ratio)
        """
        return self.slots.count/self.queue_size

    def empty(self):
        """
            Check if the queue is empty
        """
        return len(self.load_queue)==0