import simpy
import random

class Queue:
    def __init__(self, env, name, data=[], pipeline_stages=100):
        self.env = env
        self.queue = list(data)
        self.name = name
        self.inflight_push = simpy.Resource(env, capacity=pipeline_stages+1)

    def peek(self):
        print("Peeking {0} from {1}".format(self.queue[0], self.name))
        return self.queue[0]

    def pop(self):
        print("Popping {0} from {1}".format(self.queue[0], self.name))
        return self.queue.pop(0)

    def push(self, value, latency=0):
        with self.inflight_push.request() as req:
            yield req
            # Spend extra cycles (as many as the pipelining stages of the producer)
            yield self.env.timeout(latency)
            print("Inserting {0} to {1}".format(value, self.name))
            self.queue.append(value)

    def empty(self):
        return (len(self.queue) == 0)

    def done(self):
        return (self.inflight_push.count == 0) and (len(self.queue) == 0)

class DotProduct:

    def __init__(self, env, cache):

        self.env = env
        self.hint_A = Queue(env, "Hint A")
        self.hint_B = Queue(env, "Hint B")
        self.merge_queue_A = Queue(env, "Merge A")
        self.merge_queue_B = Queue(env, "Merge B")
        self.multiply_queue = Queue(env, "Multiply")
        self.accumulate_queue = Queue(env, "Accumulate")
        self.done = [False, False]
        self.merge_done = False
        self.multiply_done = False
        self.accumulate_done = False
        self.cache = cache

    def read(self, addr):
        """Read from cache"""
        # Takes 1 cycle
        yield self.env.timeout(1)
        return self.cache[addr]

    def write(self, addr, value):
        """Write to cache"""
        # Takes 0 cycle
        return self.cache[addr]

    def linear_search(self, base_idx, stream_length, value):
        """Return coord >= value"""
        cycles_spent = 0
        while(base_idx <= stream_length):
            # 1 cycle to access
            # Read the data
            data = self.cache[base_idx]
            # Return if the coord >= value
            if(data >= value):
                return base_idx, data
            else:
                base_idx += 1
            cycles_spent += 1

        yield self.env.timeout(cycles_spent)
        return None, None

    def scanner(self, start, hint_in, hint_out, merge_queue, stream_length, name=0):
        """ Scans through the stream"""
        idx = start
        coord = 0

        # Run until the end of the stream
        while(idx <= stream_length):
            # Check if we have a hint
            if(not hint_in.empty()):
                # Hint is not useful if current idx is not within the range
                hint_base, hint_bound = hint_in.peek()
                if(coord >= hint_base):
                    hint_in.pop()
                if(hint_base <= idx <= hint_bound):
                    # Move the coord ahead
                    idx, coord_next = yield self.env.process(self.linear_search(idx, stream_length, hint_bound))
                    # Reached EOS while searching
                    if(idx == None):
                        break
                else:
                    # Get the next coord and advance idx
                    coord_next = yield self.env.process(self.read(idx))

                # Add to merge queue
                self.env.process(merge_queue.push(coord_next))
                idx += 1
            else:
                # Get the next coord and advance idx
                coord_next = yield self.env.process(self.read(idx))
                idx += 1
                # Add to merge queue
                self.env.process(merge_queue.push(coord_next))

            # Send out hints (happens in the background)
            self.env.process(hint_out.push((coord, coord_next), latency=0))
            coord = coord_next

            # Tick
            yield self.env.timeout(1)

        print("Scanner {0} Done!".format(name))
        # Mark Done
        self.done[name] = True
        return None

    def merge_intersect(self, merge_queue_A, merge_queue_B, multiply_queue):
        """Merge the streams"""
        # Check if producer(s) are done and if the input Queues are empty.
        while((not all(self.done)) or ((not merge_queue_A.done()) and (not merge_queue_B.done()))):
            # Compare only if we have valid data
            if((not merge_queue_A.empty()) and (not merge_queue_B.empty())):

                ### STAGE-1 Read from Queue###

                # Peek into head
                coordA, coordB = merge_queue_A.peek(), merge_queue_B.peek()

                ### STAGE-2 Compare and add to MACC queue (if intersected)###
                # Take action accordingly
                if(coordA > coordB):
                    merge_queue_B.pop()
                elif(coordB > coordA):
                    merge_queue_A.pop()
                else:
                    merge_queue_A.pop()
                    merge_queue_B.pop()
                    # 2 Stage Pipeline
                    self.env.process(multiply_queue.push((coordA, coordB), latency=2))

            # Tick
            yield self.env.timeout(1)

        print(self.merge_queue_A.queue, self.merge_queue_B.queue)
        print("Merge Done!")
        self.merge_done = True

    def multiply(self, multiply_queue, accumulate_queue):
        """Multiply"""
        print("Multiply Instantiated")
        # Check if producer(s) are done and if the input Queues are empty.
        while((not self.merge_done) or (not multiply_queue.done())):
            if(not multiply_queue.empty()):
                # STAGE-1: Pop from queue (1 cycle)
                coords = multiply_queue.pop()
                # STAGE-2: Read the data (1 cycle)
                # Dummy data read
                # Stage-3: Multiply (1 cycle)
                dummy_result = 1
                # Insert to output queue
                self.env.process(self.accumulate_queue.push(dummy_result, latency=3))

            # Tick
            yield self.env.timeout(1)

        print("Multiply DONE!")
        self.multiply_done = True

    def accumulate(self, accumulate_queue):
        """ Accumulate"""
        print("Accumulate Instantiated")
        # Check if producer(s) are done and if the input Queues are empty.
        while((not self.multiply_done) or (not accumulate_queue.done())):
            if(not accumulate_queue.empty()):
                # Pop from queue (1 cycle)
                value = accumulate_queue.pop()
                self.result += value

            # Tick
            yield self.env.timeout(1)
        print("Accumulate DONE!")
        self.accumulate_done = True
    def execute(self, bases, bounds):

        # Set to 0
        self.result = 0
        self.done = [False, False]
        self.merge_done = False
        self.multiply_done = False
        self.accumulate_done = False

        # Start Time
        start = self.env.now

        # Instantiate two scanners and one merge
        scanner1 = self.env.process(self.scanner(bases[0], self.hint_A, self.hint_B, self.merge_queue_A, bounds[0], 0))
        scanner2 = self.env.process(self.scanner(bases[1], self.hint_B, self.hint_A, self.merge_queue_B, bounds[1], 1))
        merge = self.env.process(self.merge_intersect(self.merge_queue_A, self.merge_queue_B, self.multiply_queue))
        multiply = self.env.process(self.multiply(self.multiply_queue, self.accumulate_queue))
        accumulate = self.env.process(self.accumulate(self.accumulate_queue))

        # Run all of them until completion
        yield simpy.AllOf(self.env, [scanner1, scanner2, merge, multiply, accumulate])
        # Completion Time
        end = self.env.now

        # Completed
        print("\n\n\nCompleted in {1} Cycles! Macc Result = {0}\n\n".format(self.result, end-start))

if __name__ == "__main__":

    # Prepare
    density = 0.5
    dimension = int(1e2)
    fiber1_coords = [i for i in range(dimension) if(random.uniform(0, 1) < density)]
    fiber2_coords = [i for i in range(dimension) if(random.uniform(0, 1) < density)]

    # Preload to memory
    cache = fiber1_coords + fiber2_coords
    bases = 0, len(fiber1_coords)
    bounds = len(fiber1_coords)-1, len(cache)-1

    print("intersecting Streams: \n{0} \n{1}".format(fiber1_coords, fiber2_coords))
    # Create and launch
    env = simpy.Environment()
    dot = DotProduct(env, cache)
    proc = env.process(dot.execute(bases, bounds))

    # Run until completion
    env.run(proc)

    # Verify result
    ref_result = len(list(set(fiber1_coords) & set(fiber2_coords)))

    if(ref_result == dot.result):
        print("\nResult Verified!")
    else:
        print("Expected {0} but got {1}".format(ref_result, dot.result))