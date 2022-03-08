from copy import deepcopy
from math import prod
import numpy as np

from fate.utils.utils import truncate_idx
class Logger:

    def __init__(self, parameters) -> None:
    
        self.parameters = parameters
        self.cache_logger = {'read_hits':0, 'read_misses':0, 'write_hits':0, 'write_misses':0, 'queue_occupancy':[]}
        self.network_logger = {'read_bytes_transferred':0, 'write_bytes_transferred':0, 'utilization':[]}
        cycles_spent = {'POLL':0, 'STAGE':0, 'EXECUTE':0, 'DONE':0}
        execution_breakdown = {'sidecar':0, 'producer':0, 'consumer':0, 'compute':0}
        self.pe_logger = {  'cycles': 0,
                            'cycles_spent': deepcopy(cycles_spent),
                            'execution': deepcopy(execution_breakdown),
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

    def postprocess(self, show=True):
        """Process the self.logger values."""
        total_cycles = self.logger['end_sim'] - self.logger['start_sim']
        time = round((total_cycles/self.parameters.FREQUENCY) * 1e6, 2)
        print("Total Time: {0} ms".format(time))

        pe_logger = self.logger['PE']
        pe_wise_cycles = [pe_logger[i]['cycles']for i in range(self.parameters.NUM_PE)]

        # L1-DCache and L2 (Private. Results are weighted by cycles run)
        l1_hit_rate = [float(pe_logger[i]['L1_dCache']['read_hits'])/max(1, (pe_logger[i]['L1_dCache']['read_hits'] + pe_logger[i]['L1_dCache']['read_misses'])) for i in range(self.parameters.NUM_PE)]
        average_l1_hit_rate = 100*np.average(l1_hit_rate, weights=pe_wise_cycles)
        l2_hit_rate = [float(pe_logger[i]['L2_Cache']['read_hits'])/max(1,(pe_logger[i]['L2_Cache']['read_hits'] + pe_logger[i]['L2_Cache']['read_misses'])) for i in range(self.parameters.NUM_PE)]
        average_l2_hit_rate = 100*np.average(l2_hit_rate, weights=pe_wise_cycles)

        # L2 Cache
        l3_hitrate = 100*float(self.logger['SIDECAR_CACHE']['read_hits'])/(self.logger['SIDECAR_CACHE']['read_hits'] + self.logger['SIDECAR_CACHE']['read_misses'])

        print("L1-sidecar (Private) hitrate: {0} (dev {1}) \nL2 Sidecar (Private) Hitrate: {2} (dev {3}), \n Global Sidecar (Shared) Hitrate: {4} \n".\
            format(average_l1_hit_rate, np.std(l1_hit_rate), average_l2_hit_rate, np.std(l2_hit_rate), l3_hitrate))

        # Figure out where the time was spent in the PE
        poll_time = [pe_logger[i]['cycles_spent']['POLL'] for i in range(self.parameters.NUM_PE)]
        poll_time = np.average(poll_time, weights=pe_wise_cycles)
        stage_time = [pe_logger[i]['cycles_spent']['STAGE'] for i in range(self.parameters.NUM_PE)]
        stage_time = np.average(stage_time, weights=pe_wise_cycles)
        execute_time = [pe_logger[i]['cycles_spent']['EXECUTE'] for i in range(self.parameters.NUM_PE)]
        execute_time = np.average(execute_time, weights=pe_wise_cycles)
        total_time = poll_time + stage_time + execute_time
        print("Each PE time division: Poll: {0}%, Stage: {1}%, execute: {2}%\n ".\
            format((poll_time*100)/total_time, (stage_time*100)/total_time, (execute_time*100)/total_time ))

        # Figure out where the time was spent in the process
        consumer_time = np.average([pe_logger[i]['execution']['consumer'] for i in range(self.parameters.NUM_PE)], weights=pe_wise_cycles)
        producer_time = np.average([pe_logger[i]['execution']['producer'] for i in range(self.parameters.NUM_PE)], weights=pe_wise_cycles)
        sidecar_time = np.average([pe_logger[i]['execution']['sidecar'] for i in range(self.parameters.NUM_PE)], weights=pe_wise_cycles)
        compute_time = np.average([pe_logger[i]['execution']['compute'] for i in range(self.parameters.NUM_PE)], weights=pe_wise_cycles)
        total_time = consumer_time + producer_time + sidecar_time + compute_time
        time_division = [round((t*100)/total_time,2) for t in [consumer_time, producer_time, sidecar_time, compute_time]]
        print("Each Functional Unit time Division: Consumer: {0}%, Producer: {1}%, SideCar: {2}%, and Compute: {3}%\n".format(*time_division))

        ########## On Chip Network (PE-PE) ##########

        # Utilization

        occupancy_time, occupancy_val = list(zip(*self.logger['PE_PE_NoC']['utilization']))[0], list(zip(*self.logger['PE_PE_NoC']['utilization']))[1]
        # Need to truncaet based on time
        post_idx = truncate_idx(occupancy_time,self.logger['end_sim'])
        pre_idx = truncate_idx(occupancy_time,self.logger['start_sim'])
        average_onchip_noc_usage = round(100*np.mean(occupancy_val[pre_idx:post_idx]),2)
        
        # Bytes transferred
        total_bytes_onchip = self.logger['PE_PE_NoC']['read_bytes_transferred'] + self.logger['PE_PE_NoC']['write_bytes_transferred']
        read_bytes_onchip = 100*float(self.logger['PE_PE_NoC']['read_bytes_transferred'])/total_bytes_onchip

        # print("PE-PE NoC Utilization: {0}%, Read Traffic: {1}%".format(average_onchip_noc_usage, read_bytes_onchip))
        print("PE-PE NoC Utilization: {0}%".format(average_onchip_noc_usage))

        ########## On Chip Network (PE-SideCar) ##########
        # Utilization

        occupancy_time, occupancy_val = list(zip(*self.logger['PE_SIDECAR_NoC']['utilization']))[0], list(zip(*self.logger['PE_SIDECAR_NoC']['utilization']))[1]
        # Need to truncaet based on time
        post_idx = truncate_idx(occupancy_time,self.logger['end_sim'])
        pre_idx = truncate_idx(occupancy_time,self.logger['start_sim'])
        average_onchip_noc_usage = round(100*np.mean(occupancy_val[pre_idx:post_idx]),2)
        
        # Bytes transferred
        total_bytes_onchip = self.logger['PE_SIDECAR_NoC']['read_bytes_transferred'] + self.logger['PE_SIDECAR_NoC']['write_bytes_transferred']
        read_bytes_onchip = 100*float(self.logger['PE_SIDECAR_NoC']['read_bytes_transferred'])/total_bytes_onchip

        # print("PE-SideCar NoC Utilization: {0}%, Read Traffic: {1}%".format(average_onchip_noc_usage, read_bytes_onchip))
        print("PE-SideCar NoC Utilization: {0}%".format(average_onchip_noc_usage))

        ########## L3-DRAM NoC (Off Chip Network) ##########

        # Utilization

        occupancy_time, occupancy_val = list(zip(*self.logger['SIDECAR_DRAM_NoC']['utilization']))[0], list(zip(*self.logger['SIDECAR_DRAM_NoC']['utilization']))[1]
        # Need to truncaet based on time
        post_idx = truncate_idx(occupancy_time,self.logger['end_sim'])
        pre_idx = truncate_idx(occupancy_time,self.logger['start_sim'])
        average_SIDECAR_DRAM_NoC_usage = round(100*np.mean(occupancy_val[pre_idx:post_idx]),2)
        
        # Bytes transferred
        total_bytes_l3_dram = self.logger['SIDECAR_DRAM_NoC']['read_bytes_transferred'] + self.logger['SIDECAR_DRAM_NoC']['write_bytes_transferred']
        read_bytes_l3_dram = 100*float(self.logger['SIDECAR_DRAM_NoC']['read_bytes_transferred'])/total_bytes_l3_dram

        # print("SideCar-DRAM NoC Utilization: {0}%, Read Traffic: {1}%".format(average_SIDECAR_DRAM_NoC_usage, read_bytes_l3_dram))
        print("SideCar-DRAM NoC Utilization: {0}%".format(average_SIDECAR_DRAM_NoC_usage))