from copy import deepcopy

class Logger:

    def __init__(self, parameters) -> None:
    
        self.cache_logger = {'read_hits':0, 'read_misses':0, 'write_hits':0, 'write_misses':0, 'queue_occupancy':[]}
        self.network_logger = {'read_bytes_transferred':0, 'write_bytes_transferred':0, 'utilization':[]}
        self.pe_logger = {'num_instr':0, 'engine_busy':0, 'engine_idle':0, 'idle':0, 'total_cycles':0, 'access_latency':[], 'end_sim':0,\
                            'PE_L1_NoC': deepcopy(self.network_logger), 
                            'L1_L2_NoC': deepcopy(self.network_logger), 
                            'L1_dCache': deepcopy(self.cache_logger),
                            'L2_Cache': deepcopy(self.cache_logger)
                            }

        self.logger = {}
        self.logger['PE'] = [deepcopy(self.pe_logger) for _ in range(parameters.NUM_PE)]
        self.logger['SIDECAR_CACHE'] = deepcopy(self.cache_logger)
        self.logger['PE_SIDECAR_NoC'] = deepcopy(self.network_logger)
        self.logger['PE_PE_NoC'] = deepcopy(self.network_logger)
        self.logger['SIDECAR_DRAM_NoC'] = deepcopy(self.network_logger)
        self.logger['start_sim'] = 0
        self.logger['end_sim'] = 0