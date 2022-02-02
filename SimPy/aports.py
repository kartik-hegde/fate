import simpy

class Queue:
    def __init__(self, env, name, data=[]):
        self.env = env
        self.queue = data
        self.name = name
        
    def peek(self):
        if(self.empty()):
            yield self.producer_proc
        assert not self.empty()
        print("Peeking {0} from {1}".format(self.queue[0], self.name))
        value, timestamp = self.queue[0]
        if(timestamp > self.env.now):
            yield self.env.timeout(timestamp-self.env.now)  
        return value

    def pop(self):
        temp = self.peek()
        _ = self.queue.pop(0)
        return temp
        print("Popping {0} from {1}".format(self.queue[0], self.name))
        return self.queue.pop(0)

    # Fix the latency in the declaration
    # Or, Collision vectors 
    def push(self, value):
        if(self.full()):
            yield self.consumer_proc
        assert not self.full()
        # Spend extra cycles (as many as the pipelining stages of the producer)
        # yield self.env.timeout(latency)
        print("Inserting {0} to {1}".format(value, self.name))
        self.queue.append((value,self.env.now+latency))

    def empty(self):
        return len(self.queue) == 0
"""
    def wakeup(self):
        yield 
"""
class DummyOperator:

    def __init__(self, env, input_data=[]):
        """Initialize"""
        self.env = env
        self.output_queue = Queue(env, name="Output Queue")
        self.input_queue = Queue(env, name="Input Queue", data=input_data)
        self.stage_1_done = False
    
    def step(self, input_queue, output_queue):
        """Step-1 is first part of the operator, 3 stage pipelined"""
        # With A-ports, we will not need to model 3 stage pipeline here.
        while not input_queue.empty():
            # Stage 1: Read data
            data = input_queue.pop()
            # stage 2: Do some processing
            data *= 1
            # stage 3: Write the data out to output queue
            proc = self.env.process(output_queue.push(data, latency=3))
            # Tick
            # Can we tell simpy to just advance my clock? DOn't want to yield yet (unless queue is empty etc.)
            # Can we have a local time?
            yield self.env.timeout(1)

        self.stage_1_done = True

        return None

"""
    if(full)
        yield to consumer
    elif(empty)
        yield to producer
    else
        producer & consumer can operate
"""
    def run(self, input_data):
        """Run the operator"""
        # Create Input Queue
        self.input_queue = Queue(self.env, data=input_data, name="Input Queue")
        # Instantiate Operator Step
        step = self.env.process(self.step(self.input_queue, self.output_queue))
        self.stage_1_done = False
        yield self.env.timeout(1)

        # How to make sure you don't wake up for until that happens (yield until)

        # Run until producer done
        while((not self.stage_1_done) or (not self.output_queue.empty())):
            print("\n\n----------- Cycle {0} ----------".format(self.env.now))
            # Check if output queue is empty
            if(self.output_queue.empty()):
                print("\n\t ** Read Data = Empty\n\n")
            # Data in output queue
            else:
                print("\n\t ** Read Data = {0}\n\n".format(self.output_queue.pop()))
            yield self.env.timeout(1)

        print("\n\n------------ Complete -------------\n\n")

if __name__ == "__main__":

    # Create random data
    data = list(range(10))

    # Run the operator
    env = simpy.Environment()
    operator = DummyOperator(env)
    proc = env.process(operator.run(data))

    # Run until completion
    print("Run Starts for data: ", data)
    env.run(proc)
