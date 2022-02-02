"""
    Defines the instruction set of the processing element.
"""
import sys

# FATE IR
ISA = {
        # Loop Instructions
        'loop'      : {'type':'loop',   'heuristic':1, 'num_operands':3},
        'lend'      : {'type':'loop',   'heuristic':1, 'num_operands':3},
        'while'     : {'type':'loop',   'heuristic':1, 'num_operands':0},
        'whileend'  : {'type':'loop',   'heuristic':1, 'num_operands':0},

        # Go to a specific address (relative to base)
        'goto'      : {'type':'goto',   'heuristic':1, 'num_operands':0},

        # Branch Instructions
        'beq'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bneq'      : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bg'        : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bng'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bge'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bl'        : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bnl'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'ble'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bal'       : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'bunal'     : {'type':'branch',   'heuristic':1, 'num_operands':0},
        'balr'      : {'type':'branch',   'heuristic':1, 'num_operands':0},

        # Conditional Instructions that operate of regs (set flags)
        'cmp'       : {'type':'cond',   'heuristic':1, 'num_operands':2},
        'gt'        : {'type':'cond',   'heuristic':1, 'num_operands':2},
        'ge'        : {'type':'cond',   'heuristic':1, 'num_operands':2},
        'lt'        : {'type':'cond',   'heuristic':1, 'num_operands':2},
        'le'        : {'type':'cond',   'heuristic':1, 'num_operands':2},

        # Arith instructions Reg-Reg
        'mov'       : {'type':'reg',    'heuristic':1, 'num_operands':2},
        'swp'       : {'type':'reg',    'heuristic':1, 'num_operands':2},
        'add'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'addi'      : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'mac'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'sub'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'subf'      : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'div'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'ceil'      : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'max'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'min'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'clear'     : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'mul'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'lsl'       : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'addaligned': {'type':'reg',    'heuristic':1, 'num_operands':3},
        'addsi'     : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'movvr'     : {'type':'reg',    'heuristic':1, 'num_operands':3},
        'cswp'      : {'type':'reg',    'heuristic':1, 'num_operands':3},

        # Vector Instructions
        'vmov'      : {'type':'vector', 'heuristic':1, 'num_operands':2},
        'vadd'      : {'type':'vector', 'heuristic':1, 'num_operands':3},
        'vsub'      : {'type':'vector', 'heuristic':1, 'num_operands':3},
        'vmul'      : {'type':'vector', 'heuristic':1, 'num_operands':3},

        # Sync Instructions
        'poll'      : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'acquire'   : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'release'   : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'aread'     : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'aupdate'   : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'aincr'     : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'wait'      : {'type':'sync',   'heuristic':1, 'num_operands':3},
        'barrier'   : {'type':'sync',   'heuristic':1, 'num_operands':0},

        # Load
        'vload'     : {'type':'vload',  'heuristic':0, 'num_operands':2},
        'vloadi'    : {'type':'vload',  'heuristic':0, 'num_operands':2},
        'vstore'    : {'type':'vstore', 'heuristic':1, 'num_operands':2},
        'vstorei'   : {'type':'vstore', 'heuristic':1, 'num_operands':2},
        'load'      : {'type':'load',   'heuristic':0, 'num_operands':2},
        'loadi'     : {'type':'load',   'heuristic':0, 'num_operands':2},
        'dload'     : {'type':'load',   'heuristic':0, 'num_operands':2},
        'dloadi'    : {'type':'load',   'heuristic':0, 'num_operands':2},

        # Blocked loads (these will be executed in order, no strands
        'vbload'    : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'vbloadi'   : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'dvbload'   : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'dvbloadi'  : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'bload'     : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'bloadi'    : {'type':'bload',  'heuristic':1, 'num_operands':2},
        'dbloadi'   : {'type':'bload',  'heuristic':1, 'num_operands':2},

        # Store
        'store'     : {'type':'store',  'heuristic':1, 'num_operands':2},
        'storei'    : {'type':'storei', 'heuristic':1, 'num_operands':2},
        'dstore'    : {'type':'dstore',  'heuristic':1, 'num_operands':2},
        'dstorei'   : {'type':'dstorei', 'heuristic':1, 'num_operands':2},

        # Special Purpose Instructions
        'neighbors' : {'type':'special', 'heuristic':1, 'num_operands':3},
        'unvisited' : {'type':'special', 'heuristic':1, 'num_operands':3},

}

class Instuction:
    """ All Instructions are of this object type"""
    def __init__(self, instr, op_type, operands, heuristic):
        self.instr = instr
        self.op_type = op_type
        self.operands = operands
        self.heuristic = heuristic

class Operand:
    """ Each operand is this object."""
    def __init__(self, operand_type, operand_value, single_use):
        self.operand_type = operand_type
        self.operand_value = operand_value
        self.single_use = single_use

def decode(node):
    """
        Given a node of a graph, this decodes it.
    """
    string = node.payload.rstrip().split(" ")

    # Decode
    instr = string[0]
    operands = string[1:]
    operands_decoded = []

    # Update the type and value of operands.
    for operand in operands:
        # Register

        # Check if the operand is single use
        single_use = (operand[-1] == '!')
        if(single_use):
            operand = operand[:-1]

        # Decode
        if(operand[0] == 'R'):
            operand_type, operand_value = 'register', int(operand[1:])
        elif(operand[0] == 'V'):
            operand_type, operand_value = 'vector', int(operand[1:])
        elif(operand.isdigit() or ('0x' in operand) or (operand[0] == '-' and operand[1:].isdigit())):
            operand_type, operand_value = 'immediate', int(operand, 16) if('0x' in operand) else int(operand)
        else:
            # May be is a float value, try.
            try:
                float(operand)
            except ValueError:
                sys.exit("Operand {0} is of unsupported type for instr {1}.".format(operand, instr))

        operands_decoded.append(Operand(operand_type,operand_value, single_use))

    decoded_instr = ISA[instr]

    return Instuction(instr, decoded_instr['type'], operands_decoded, decoded_instr['heuristic'])