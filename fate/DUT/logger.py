from copy import deepcopy
from math import prod
import numpy as np

from fate.utils.utils import truncate_idx, smoothen, softmax, average, write_csv
from fate.utils.plot_graph import plot_graph, plot_graph_double_y, plot_graph_bar

class Logger:

    def __init__(self, parameters) -> None:
    
        self.parameters = parameters
        self.cache_logger = {'read_hits':0, 'read_misses':0, 'write_hits':0, 'write_misses':0, 'queue_occupancy':[]}
        self.network_logger = {'read_bytes_transferred':0, 'write_bytes_transferred':0, 'utilization':[]}
        self.buffet_logger = {'name': None, 'production':[], 'consumption':[], 'usage':[]}
        cycles_spent = {'POLL':0, 'STAGE':0, 'EXECUTE':0, 'DONE':0}
        execution_breakdown = {'RoP':[], 'sidecar':0, 'producer':0, 'consumer':0, 'compute':0}
        self.pe_logger = {  'name' : None,
                            'cycles': 0,
                            'cycles_spent': deepcopy(cycles_spent),
                            'execution': deepcopy(execution_breakdown),
                            'PE_L1_NoC': deepcopy(self.network_logger), 
                            'L1_L2_NoC': deepcopy(self.network_logger), 
                            'L1_dCache': deepcopy(self.cache_logger),
                            'L2_Cache': deepcopy(self.cache_logger),
                            'buffet_logger': deepcopy(self.buffet_logger),
                            'buffets': []
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

        # L1-DCache and L2 (Private. Results)
        l1_hit_rate = [float(pe_logger[i]['L1_dCache']['read_hits'])/max(1, (pe_logger[i]['L1_dCache']['read_hits'] + pe_logger[i]['L1_dCache']['read_misses'])) for i in range(self.parameters.NUM_PE)]
        average_l1_hit_rate = 100*np.average(l1_hit_rate, weights=pe_wise_cycles)
        l2_hit_rate = [float(pe_logger[i]['L2_Cache']['read_hits'])/max(1,(pe_logger[i]['L2_Cache']['read_hits'] + pe_logger[i]['L2_Cache']['read_misses'])) for i in range(self.parameters.NUM_PE)]
        average_l2_hit_rate = 100*np.average(l2_hit_rate, weights=pe_wise_cycles)

        # L3 Cache
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


        ########## Producer-Consumer Relationship ##########
        print("\n\n")
        buffet_loggers = []
        prodcon_data = [['Buffet Name', 'Production Rate (cycles)', 'Consumption Rate (cycles)', 'Avg Utilization (%)']]
        for pe in range(self.parameters.NUM_PE):
            logger = self.logger['PE'][pe]
            buffet_loggers += logger['buffets']

            # Rate of production and consumption
            production_x = []
            production_y = []
            consumption_x = []
            consumption_y = []
            legends = []
            # Gather all the data
            for buffet_logger in logger['buffets']:
                buffet_name = buffet_logger['name']
                production_rate = np.diff(buffet_logger['production'], prepend=0)
                consumption_rate = np.diff(buffet_logger['consumption'], prepend=0)
                utilization = list(zip(*buffet_logger['usage']))[1]
                # Save the data to plot
                legends.append(buffet_name)
                production_x.append(np.array(buffet_logger['production']))
                production_y.append(smoothen(production_rate))
                consumption_x.append(np.array(buffet_logger['consumption']))
                consumption_y.append(smoothen(consumption_rate))
                print("\nBuffet {0}. Rate of Production: {1}, Rate of Consumption: {2}, Avg Occupancy: {3}%".\
                    format(buffet_name, round(average(production_rate),2), round(average(consumption_rate),2), round(average(utilization),2)))
                prodcon_data.append([buffet_name, round(average(production_rate),2), round(average(consumption_rate),2), round(average(utilization),2)])
                # plot_graph([production_x, consumption_x], [production_y, consumption_y], labels=('Time', 'Cycles'), plot_title='Production vs Consumption', file_name='plots/production_rate_' + str(buffet_name))
            # name = logger['name']
            # rop_data = logger['execution']['RoP']
            # if(rop_data):
            #     rop, occupancy, times = list(zip(*rop_data))[0], list(zip(*rop_data))[1], list(zip(*rop_data))[2]
            #     rop = smoothen(rop)
            #     occupancy = smoothen(occupancy)
            #     plot_graph_double_y([times,], [rop,], [occupancy,], labels=('Cycles', 'Rate of Production', "Occupancy (%)"), plot_title='Rate of Production',file_name='plots/occupancy_regfile_'+str(name))
            write_csv(prodcon_data, 'plots/prodcondata.csv')
            print("\n\n")
            
        # Figure out where the time was spent in the process
        series = []
        series_labels = []

        consumer_time = [pe_logger[i]['execution']['consumer'] for i in range(self.parameters.NUM_PE)]
        producer_time = [pe_logger[i]['execution']['producer'] for i in range(self.parameters.NUM_PE)]
        sidecar_time = [pe_logger[i]['execution']['sidecar'] for i in range(self.parameters.NUM_PE)]  
        compute_time = [pe_logger[i]['execution']['compute'] for i in range(self.parameters.NUM_PE)]
        for i in range(self.parameters.NUM_PE):
            division = [t[i] for t in [consumer_time, producer_time, sidecar_time, compute_time]]
            series_labels.append(pe_logger[i]['name'])
            division = softmax(division)
            series.append(np.array(division))
            print("\nPE with Payload {4} spent times as: Consumer: {0}%, Producer: {1}%, SideCar: {2}%, and Compute: {3}%\n".format(*division,pe_logger[i]['name']))
        time_division = softmax([np.average(t, weights=pe_wise_cycles) for t in [consumer_time, producer_time, sidecar_time, compute_time]])
        print("Each Functional Unit time Division: Consumer: {0}%, Producer: {1}%, SideCar: {2}%, and Compute: {3}%\n".format(*time_division))
        # To plot them, convert to numpy arrays
        series = np.array(series)
        plot_graph_bar(np.transpose(series), series_labels, legends=['consumer', 'producer', 'sidecar', 'compute'], plot_labels=['Percentage', 'Operators'], path_to_save='/plots/operator_bottlenecks')