import queue
import simpy

class SimpleQueue:
    def __init__(self, name, data=[]):
        self.queue = list(data)
        self.name = name

    def peek(self):
        print("Peeking {0} from {1}".format(self.queue[0], self.name))
        return self.queue[0]

    def pop(self):
        print("Popping {0} from {1}".format(self.queue[0], self.name))
        return self.queue.pop(0)

    def push(self, value, latency=0):
        print("Inserting {0} to {1}".format(value, self.name))
        self.queue.append(value)

    def empty(self):
        return (len(self.queue) == 0)

    def done(self):
        return (self.inflight_push.count == 0) and (len(self.queue) == 0)

class SimpyQueue:
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