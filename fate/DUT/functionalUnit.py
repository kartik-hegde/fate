"""
    Define functional Units.
"""
from concurrent.futures import process
import simpy
import math
import sys

class Operator:
    def __init__(self, run_function, operands) -> None:
        self.run_function = run_function
        self.operands = operands

class FunctionalUnit:

    def __init__(self, env, parameters, name, l1d_cache, logger) -> None:
        self.env = env
        self.parameters = parameters
        self.name = name
        self.sidecar = l1d_cache
        self.logger = logger
        self.operators = {
            'GetHandles': {'type': '', 'function':self.GetHandles, 'num_operands': 1,},
            'HandlesToCoords': {'type': '', 'function':self.HandlesToCoords, 'num_operands': 1,},
            'HandlesToValues': {'type': '', 'function':self.HandlesToValues, 'num_operands': 1,},
            'Intersect': {'type': '', 'function':self.Intersect, 'num_operands': 1,},
            'Compute': {'type': '', 'function':self.Compute, 'num_operands': 1,},
            'Populate': {'type': '', 'function':self.Populate, 'num_operands': 1,},
        }

    def decode(self, instruction):
        """Decode the payload"""
        decoded = instruction.rstrip().split(" ")
        decoded_operator, operands = decoded[0], decoded[1:]
        operator_dict = self.operators[decoded_operator]

        return Operator(operator_dict['function'], operands)

    def run(self, payload, args):
        """Execute the payload"""
        # Decode
        operator = self.decode(payload)
        # Launch the appropriate function
        yield self.env.process(operator.run_function(operator, args))

    def fill_consumers(self, stream_value, consumers):
        """
            This is a helper function that writes one stream to many consumers in parallel.

            This is a broadcast operation. Same value is written to many streams.
        """
        procs = []
        for consumer in consumers:
            procs.append(self.env.process(consumer.fill(stream_value)))
        # Run them all in parallel and save the logs.
        yield simpy.events.AllOf(self.env, procs)

    def fill_all_consumers(self, streams, consumers):
        """
            This is a helper function that writes to all streams and all consumers in parallel.

            This is a multicast operation. streams is a vector of size N, consumers is a matrix of size NxM.
        """
        procs = []
        for stream_id, stream_value in enumerate(streams):
             for consumer in consumers[stream_id]:
                 procs.append(self.env.process(consumer.fill(stream_value)))
        # Run them all in parallel.
        yield simpy.events.AllOf(self.env, procs)

    def read_stream(self, stream):
        """Read from a stream"""
        val = yield self.env.process(stream.read(0, shrink=True))
        return val

    def read_streams(self, streams):
        """Read from many streams in parallel"""
        procs = [self.env.process(self.read_stream(stream)) for stream in streams]
        # Run them all in parallel.
        values = yield simpy.events.AllOf(self.env, procs)
        # print(values, self.name)
        # input()
        return list(values.values())
    
    def drain_stream(self, stream):
        """Helper function to strain a stream"""
        data = None
        while(data != 'EOS'):
            data = yield self.env.process(stream.read(0, shrink=True))

    def drain_streams(self, streams):
        """Helper function to strain a stream"""
        procs = [self.env.process(self.drain_stream(stream)) for stream in streams]
        # Run them all in parallel.
        yield simpy.events.AllOf(self.env, procs)

    # Stat vs Event (give more info)
    def GetHandles(self, operator, args):
        """
            Function: Get the handles of all (coord, value) pairs in the fiber

            Inputs:
            --------
            1. Fiber Handle (sidecar)

            Outputs:
            ________
            1. Handles for every non-zero element ((coord,val) pair)
            
        """
        # TODO: Streaming Memory --> Channel
        # Extract arguments
        _, [streaming_memories_out] = args
        fiber_base = int(operator.operands[0])

        start_time = self.env.now
        # Step 1: Read from side-car (TODO: nnz --> Occupancy)
        nnz_count = yield self.env.process(self.sidecar.read(fiber_base))
        self.logger['sidecar'] += self.env.now - start_time

        # Step 2: Create handles and send results out
        # Each word is 32-bit. Each handle points to (coord, val) pair, hence 2 words.
        # TODO: Change from ElementPointer --> ElementHandle
        # TODO: EOS needs tags

        start_time = self.env.now
        # For every output memory, send out the results. (There is only one output stream.)
        for idx in range(nnz_count+1):
            handle = fiber_base + ((1 + (idx * 2))<<2) if(idx < nnz_count) else 'EOS'
            yield self.env.process(self.fill_consumers(handle, streaming_memories_out))
            # Logging statistics
            self.logger['consumer'] += self.env.now - start_time - 1
            self.logger['compute'] += 1
            start_time = self.env.now

        print("GetHandles Complete at {0} at {1}.\n".format(self.name, self.env.now))


    def HandlesToCoords(self, operator, args):
        """
            Function: Given handles, get coords

            Inputs:
            --------
            1. Fiber Handle (sidecar)
            2. Handles (stream)

            Outputs:
            ________
            1. Coords for every non-zero element (stream)
            
        """
        # Extract arguments
        [Handles], [streaming_memories_out] = args
        # TODO : Use fiber_base to offset in instead of using actual pointers
        fiber_base = int(operator.operands[0])
        # For every handle, get the coords and send out
        while True:
            ## ** Producer ** ##
            # Logging statistics
            start_time = self.env.now
            # Read from stream in
            handle =  yield self.env.process(Handles.read(0, shrink=True)) # size instead of true?
            self.logger['producer'] += self.env.now - start_time

            ## ** Sidecar ** ##
            # Logging statistics
            start_time = self.env.now
            # If reached EOS, break
            if(handle == 'EOS'): # TODO: multiple levels of streams (Max levels of streams?)
                # Write to stream out
                yield self.env.process(self.fill_consumers('EOS', streaming_memories_out))
                break
            # Read from sidecar
            coord = yield self.env.process(self.sidecar.read(handle))
            self.logger['sidecar'] += self.env.now - start_time

            ## ** Consumer ** ##
            # Logging statistics
            start_time = self.env.now
            # Write to stream out
            yield self.env.process(self.fill_consumers(coord, streaming_memories_out))
            self.logger['consumer'] += self.env.now - start_time - 1
            self.logger['compute'] += 1

        print("HandlesToCoords Complete at {0} at {1}.\n".format(self.name, self.env.now))

    def HandlesToValues(self, operator, args):
        """
            Function: Given handles, get values

            Inputs:
            --------
            1. Fiber Handle (sidecar)
            2. Handles (stream)

            Outputs:
            ________
            1. Values for every non-zero element ((coord,val) pair)
            
        """
        # Extract arguments
        [Handles], [streaming_memories_out] = args
        fiber_base = int(operator.operands[0])
        # For every handle, get the vals and send out
        while True:

            ## ** Producer ** ##
            # Logging statistics
            start_time = self.env.now
            # Read from stream in
            handle =  yield self.env.process(Handles.read(0, shrink=True)) # size instead of true?
            self.logger['producer'] += self.env.now - start_time

            ## ** Sidecar ** ##
            # Logging statistics
            start_time = self.env.now
            # If reached EOS, break
            if(handle == 'EOS'): # TODO: multiple levels of streams (Max levels of streams?)
                # Write to stream out
                yield self.env.process(self.fill_consumers('EOS', streaming_memories_out))
                break
            # Read from sidecar
            val = yield self.env.process(self.sidecar.read(handle+ (1 << 2)))
            self.logger['sidecar'] += self.env.now - start_time

            ## ** Consumer ** ##
            # Logging statistics
            start_time = self.env.now
            # Write to stream out
            yield self.env.process(self.fill_consumers(val, streaming_memories_out))
            self.logger['consumer'] += self.env.now - start_time - 1
            self.logger['compute'] += 1

        print("HandlesToValues Complete at {0} at {1}.\n".format(self.name, self.env.now))

    def Intersect(self, operator, args):
        """
            Function: Given two fibers (handles and coords), give out intersected fiber

            Inputs:
            --------
            1. Coords Stream 1 (stream)
            2. Handles Stream 1 (stream)
            3. Coords Stream 2 (stream)
            4. Handles Stream 2 (stream)

            [1,2] are isomorphic (strong), [3,4] are isomorphic (strong)
            Outputs:
            ________
            1. Coords Intersected Stream (stream)
            2. Handles Stream 1 (stream)
            3. Handles Stream 2 (stream)
            
            [1,2,3] strong isomorphism
        """
        # Extract arguments
        streaming_memories_in, streaming_memories_out = args
        [sm_in_coords_s1, sm_in_handles_s1, sm_in_coords_s2, sm_in_handles_s2],\
            [sm_out_coords_is, sm_out_handles_s1, sm_out_handles_s2] = args

        coords_s1, handles_s1, coords_s2, handles_s2 = yield self.env.process(self.read_streams(streaming_memories_in))
        # Go through streams
        while True:
            # End of stream
            if((coords_s1 == 'EOS') or (coords_s2 == 'EOS')):
                # TODO: Add assertions
                # Drain output streams. Can occur asynchronously (TODO: chcek if this causes issues)
                if(coords_s1 == 'EOS'):
                    print("\nCoords Stream on S1 ended\n")
                    yield self.env.process(self.drain_streams([sm_in_coords_s2, sm_in_handles_s2]))
                elif(coords_s2 == 'EOS'):
                    print("\nCoords Stream on S2 ended\n")
                    yield self.env.process(self.drain_streams([sm_in_coords_s1, sm_in_handles_s1]))
                # Send end of streams
                yield self.env.process(self.fill_all_consumers(['EOS', 'EOS', 'EOS'], streaming_memories_out))
                # Intersection Complete
                break
            # Match
            elif(coords_s1 == coords_s2):
                ## ** Consumer ** ##
                # Logging statistics
                start_time = self.env.now
                yield self.env.process(self.fill_all_consumers((coords_s1, handles_s1, handles_s2), streaming_memories_out))
                self.logger['consumer'] += self.env.now - start_time

                ## ** Producer ** ##
                # Logging statistics
                start_time = self.env.now
                coords_s1, handles_s1, coords_s2, handles_s2 = yield self.env.process(self.read_streams(streaming_memories_in))
                self.logger['producer'] += self.env.now - start_time - 1
                self.logger['compute'] += 1

            elif(coords_s1 > coords_s2):
                ## ** Producer ** ##
                # Logging statistics
                start_time = self.env.now
                coords_s2, handles_s2 = yield self.env.process(self.read_streams([sm_in_coords_s2, sm_in_handles_s2]))
                self.logger['producer'] += self.env.now - start_time - 1
                self.logger['compute'] += 1
            else:
                ## ** Producer ** ##
                # Logging statistics
                start_time = self.env.now
                coords_s1, handles_s1 = yield self.env.process(self.read_streams([sm_in_coords_s1, sm_in_handles_s1]))
                self.logger['producer'] += self.env.now - start_time - 1
                self.logger['compute'] += 1

        print("Intersect Complete at {0} at {1}.\n".format(self.name, self.env.now))
    
    def Compute(self, operator, args):
        """
            Function: Given two streams of values, Compute on them

            Inputs:
            --------
            1. Values Stream 1 (stream)
            2. Values Stream 2 (stream)

            Outputs:
            ________
            1. Values Computed Stream (stream)
            
        """
        # Extract Arguments
        streaming_memories_in, streaming_memories_out = args
        [sm_in_values_s1, sm_in_values_s2], [sm_out_values] = args
        compute_operator = operator.operands[0]
        # Go through stream
        while True:
            ## ** Producer ** ##
            # Logging statistics
            start_time = self.env.now
            values_s1, values_s2 = yield self.env.process(self.read_streams(streaming_memories_in))
            self.logger['producer'] += self.env.now - start_time 

            # Guaranteed to be isomorphic
            if(values_s1 == 'EOS'):
                # End of stream
                yield self.env.process(self.fill_consumers('EOS', sm_out_values))
                break
            if(compute_operator == 'add'):
                result = values_s1 + values_s2
            elif(compute_operator == 'mul'):
                result = values_s1 * values_s2
            else:
                sys.exit("Compute Operator {0} not understood".format(compute_operator))
            # Takes a cycle to compute #TODO: parameterize
            yield self.env.timeout(1)

            ## ** Consumer ** ##
            # Logging statistics
            start_time = self.env.now
            # Write output
            yield self.env.process(self.fill_consumers(result, sm_out_values))
            self.logger['consumer'] += self.env.now - start_time - 1
            self.logger['compute'] += 1

        print("Complete {2} at {0} at {1}.\n".format(self.name, self.env.now, compute_operator))

    def Populate(self, operator, args):
        """
            Function: Given two streams of coords and values, populate and empty fiber

            Inputs:
            --------
            1. Coords Stream 1 (stream)
            2. Values Stream 1 (stream)
            3. Fiber Handle (sidecar)

            Outputs:
            ________
            Acks TODO

            TODO:
            Break into two: Get handle, and then updatevalue/updatefiberhandle
            
        """
        # Extract Arguments
        streaming_memories_in, streaming_memories_out = args
        [sm_in_coords_s1, sm_in_values_s1], [] = args
        fiber_base = int(operator.operands[0])
        # Header is reserved
        addr = fiber_base + (1 << 2)
        nnz = 0
        # Read and populate
        while True:

            ## ** Producer ** ##
            # Logging statistics
            start_time = self.env.now
            # Read from streams
            coords_s1, values_s1 = yield self.env.process(self.read_streams(streaming_memories_in))
            self.logger['producer'] += self.env.now - start_time 

            # Guaranteed to be isomorphic
            if(values_s1 == 'EOS'):
                break
            # Write to sidecar (assuming pipelined with above read, hence no extra cycles added)
            self.env.process(self.sidecar.write(addr, coords_s1))
            addr += (1<<2)
            self.env.process(self.sidecar.write(addr, values_s1))
            addr += (1<<2)
            # Increment nnz
            nnz += 1
            # Udpate Logger
            self.logger['compute'] += 1

        # Write the header
        yield self.env.process(self.sidecar.write(fiber_base, nnz))

        print("Populate Complete at {0} at {1}.\n".format(self.name, self.env.now))

