"""
    Define functional Units.
"""
import simpy
import math
import sys

class Operator:
    def __init__(self, run_function, operands) -> None:
        self.run_function = run_function
        self.operands = operands

class FunctionalUnit:

    def __init__(self, env, parameters, l1d_cache) -> None:
        self.env = env
        self.parameters = parameters
        self.sidecar = l1d_cache
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
        operator_dict = self.operators(decoded_operator)

        return Operator(operator_dict['function'], operands)

    def run(self, payload, args):
        """Execute the payload"""
        # Decode
        operator = self.decode(payload)
        # Launch the appropriate function
        yield self.env.process(operator.run_function(operator, args))

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
        # Extract arguments
        _, [streaming_memory_out] = args
        fiber_base = int(operator.operands[0])
        # Read from side-car
        nnz_count = yield self.sidecar.read(fiber_base)
        # Send header (size of stream) to stream out
        yield streaming_memory_out.fill(nnz_count)
        # Create handles
        # Each word is 32-bit. Each handle points to (coord, val) pair, hence 2 words.
        handles = [fiber_base + ((1 + (idx * 2))<<2) for idx  in range(nnz_count)]
        yield streaming_memory_out.bulk_fill(handles)
        # Send End of Stream
        yield streaming_memory_out.fill('EOS')

        
    def HandlesToCoords(self, operator, args):
        """
            Function: Given handles, get coords

            Inputs:
            --------
            1. Fiber Handle (sidecar)
            2. Handles (stream)

            Outputs:
            ________
            1. Coords for every non-zero element
            
        """
        # Extract arguments
        [streaming_memory_in], [streaming_memory_out] = args
        fiber_base = int(operator.operands[0])
        # For every handle, get the coords and send out
        while True:
            # Read from stream in
            handle =  yield streaming_memory_in.read(0, shrink=True)
            if(handle == 'EOS'):
                break
            # Read from sidecar
            coord = yield self.sidecar.read(handle)
            # Write to stream out
            yield streaming_memory_out.fill(coord)
        # Send End of Stream
        yield streaming_memory_out.fill('EOS')


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
        [streaming_memory_in], [streaming_memory_out] = args
        fiber_base = int(operator.operands[0])
        # For every handle, get the vals and send out
        while True:
            # Read from stream in
            handle =  yield streaming_memory_in.read(0, shrink=True)
            if(handle == 'EOS'):
                break
            # Read from sidecar
            val = yield self.sidecar.read(handle+ (1 << 2))
            # Write to stream out
            yield streaming_memory_out.fill(val)
        # Send End of Stream
        yield streaming_memory_out.fill('EOS')

    def Intersect(self, operator, args):
        """
            Function: Given two fibers (handles and coords), give out intersected fiber

            Inputs:
            --------
            1. Coords Stream 1 (stream)
            2. Handles Stream 1 (stream)
            3. Coords Stream 2 (stream)
            4. Handles Stream 2 (stream)

            Outputs:
            ________
            1. Coords Intersected Stream (stream)
            2. Handles Stream 1 (stream)
            2. Handles Stream 2 (stream)
            
        """
        # Extract arguments
        [sm_in_coords_s1, sm_in_handles_s1, sm_in_coords_s2, sm_in_handles_s2],\
            [sm_out_coords_is, sm_out_handles_s1, sm_out_handles_s2] = args
        # Define useful functions
        def read_s1_s2():
            coords_s1 = yield  sm_in_coords_s1.read(0, shrink=True)
            handles_s1 = yield  sm_in_handles_s1.read(0, shrink=True)
            coords_s2 = yield  sm_in_coords_s2.read(0, shrink=True)
            handles_s2 = yield  sm_in_handles_s2.read(0, shrink=True)
            return (coords_s1, handles_s1, coords_s2, handles_s2)

        def read_s1():
            coords_s1 = yield  sm_in_coords_s1.read(0, shrink=True)
            handles_s1 = yield  sm_in_handles_s1.read(0, shrink=True)
            return (coords_s1, handles_s1)

        def read_s2():
            coords_s2 = yield  sm_in_coords_s2.read(0, shrink=True)
            handles_s2 = yield  sm_in_handles_s2.read(0, shrink=True)
            return (coords_s2, handles_s2)

        def drain_stream(stream):
            data = yield stream.read(0, shrink=True)
            while(data != 'EOS'):
                data = yield stream.read(0, shrink=True)

        coords_s1, handles_s1, coords_s2, handles_s2 = yield self.env.process(read_s1_s2())
        # Go through streams
        while True:
            # End of stream
            if((coords_s1 == 'EOS') or (coords_s2 == 'EOS')):
                # Drain output streams. Can occur asynchronously (TODO: chcek if this causes issues)
                if(coords_s1 != 'EOS'):
                    self.env.process(drain_stream(coords_s1))
                    self.env.process(drain_stream(handles_s1))
                elif(coords_s2 != 'EOS'):
                    self.env.process(drain_stream(coords_s2))
                    self.env.process(drain_stream(handles_s2))
                break
            # Match
            elif(coords_s1 == coords_s2):
                yield sm_out_coords_is.fill(coords_s1)
                yield sm_out_handles_s1.fill(handles_s1)
                yield sm_out_handles_s2.fill(handles_s2)
                coords_s1, handles_s1, coords_s2, handles_s2 = yield self.env.process(read_s1_s2())
            elif(coords_s1 > coords_s2):
                coords_s2, handles_s2 = yield self.env.process(read_s2())
            else:
                coords_s1, handles_s1 = yield self.env.process(read_s1())
                
        # Send end of streams
        yield sm_out_coords_is.fill('EOS')
        yield sm_out_handles_s1.fill('EOS')
        yield sm_out_handles_s2.fill('EOS')

    
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
        [sm_in_values_s1, sm_in_values_s2], [sm_out_values] = args
        compute_operator = operator.operands[0]
        # Go through stream
        while True:
            values_s1 = yield  sm_in_values_s1.read(0, shrink=True)
            values_s2 = yield  sm_in_values_s2.read(0, shrink=True)
            # Guaranteed to be isomorphic
            if(values_s1 == 'EOS'):
                break
            if(compute_operator == 'add'):
                result = values_s1 + values_s2
            elif(compute_operator == 'mul'):
                result = values_s1 * values_s2
            else:
                sys.exit("Compute Operator {0} not understood".format(compute_operator))
            # Takes a cycle to compute
            yield self.env.timeout(1)
            # Write output
            yield sm_out_values.fill(result)
        # End of stream
        yield sm_out_values.fill('EOS')


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
            None
            
        """
        # Extract Arguments
        [sm_in_coords_s1, sm_in_values_s1], [] = args
        fiber_base = int(operator.operands[0])
        # Header is reserved
        addr = fiber_base + (1 << 2)
        nnz = 0
        # Read and populate
        while True:
            # Read from streams
            values_s1 = yield sm_in_values_s1.read(0, shrink=True)
            coords_s1 = yield sm_in_coords_s1.read(0, shrink=True)
            # Guaranteed to be isomorphic
            if(values_s1 == 'EOS'):
                break
            # Write to sidecar
            yield self.sidecar.write(addr, coords_s1)
            addr += (1<<2)
            yield self.sidecar.write(addr, values_s1)
            addr += (1<<2)
            # Increment nnz
            nnz += 1
        # Write the header
        yield self.sidecar.write(fiber_base, nnz)

        print("Populate Complete at ", self.env.now)

