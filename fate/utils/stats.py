"""
    Create a suitable logger for the accelerator to capture relevant statistics.
"""
from copy import deepcopy
import numpy as np
import os
import sys
from DDGE.utils.plot_graph import plot_graph, smoothen
import random
from collections import OrderedDict

def truncate_idx(time_series, time, is_tuple=False):
    """
        Index to truncate the series.
    """
    for idx,val in enumerate(time_series):
        if(val >= time):
            return idx
    return idx

def percentagify(lst):
    return [100*i for i in lst]

def anti_cum_sum(series):
    """
        Reverse the cumulative sum.
    """
    if(len(series)==0):
        return []
    else:
        return [0,] + [series[i+1]-series[i] for i in range(len(series)-1)]

def postprocess(logged, show=True):

    # For all the PEs
    num_pe =  1 #len(logged['PE'])
    compute_util = [logged['PE'][i]['engine_busy'] for i in range(num_pe)]
    mem_wait = [logged['PE'][i]['engine_idle'] for i in range(num_pe)]
    cycles = [logged['PE'][i]['total_cycles'] for i in range(num_pe)]
    access_latency = [sum(logged['PE'][i]['access_latency'])/len(logged['PE'][i]['access_latency']) for i in range(num_pe)]

    start_sim = logged['start_sim']
    end_sim = logged['end_sim']

    if(not os.path.isdir('plots/')):
        os.mkdir('plots/')
        os.mkdir('plots/regs')
        os.mkdir('plots/threads')

    # Register File Occupancy in each PE
    average_regfile_occupancy =[]
    for i in range(num_pe):
        occupancy_time, occupancy_val = list(zip(*logged['PE'][i]['regfile_occupancy']))[0], list(zip(*logged['PE'][i]['regfile_occupancy']))[1]
        # We will truncate the index based on the end of simulation (avoid any later points.)
        idx = truncate_idx(occupancy_time, logged['PE'][i]['end_sim'])
        average_regfile_occupancy.append(np.mean(occupancy_val[:idx]))
        # Clean out the data until the truncation
        x_series = occupancy_time[:idx]
        y_series = smoothen(percentagify(occupancy_val[:idx]))
        if(i==0 and show):
            plot_graph([x_series,], [y_series,], labels=('Cycles', "Occupancy (%)"), plot_title='Regfile Usage',file_name='plots/occupancy_regfile_'+str(i))
    average_regfile_occupancy = 100*np.average(average_regfile_occupancy, weights=cycles)

    # Thread Occupancy
    average_thread_occupancy =[]
    for i in range(num_pe):
        occupancy_time, occupancy_val = list(zip(*logged['PE'][i]['thread_occupancy']))[0], list(zip(*logged['PE'][i]['thread_occupancy']))[1]
        # We will truncate the index based on the end of simulation (avoid any later points.)
        idx = truncate_idx(occupancy_time, logged['PE'][i]['end_sim'])
        average_thread_occupancy.append(np.mean(occupancy_val[:idx]))
        # Clean out the data until the truncation
        x_series = occupancy_time[:idx]
        y_series = smoothen(occupancy_val[:idx])
        if(i==0 and show):
            plot_graph([x_series,], [y_series,], labels=('Cycles', "Number of Strands"), plot_title='Strands Active/PE',file_name='plots/occupancy_thread_'+str(i))
    average_thread_occupancy = np.average(average_thread_occupancy, weights=cycles)

    # Store Queue Occupancy
    average_lq_occupancy =[]
    for i in range(num_pe):
        occupancy_time, occupancy_val = list(zip(*logged['PE'][i]['lq_occupancy']))[0], list(zip(*logged['PE'][i]['lq_occupancy']))[1]
        # We will truncate the index based on the end of simulation (avoid any later points.)
        idx = truncate_idx(occupancy_time, logged['PE'][i]['end_sim'])
        average_lq_occupancy.append(np.mean(occupancy_val[:idx]))
        if(i==0 and show):
            plot_graph([occupancy_time[:idx],], [percentagify(occupancy_val[:idx]),], labels=('Cycles', "Load Queue Occupancy"), plot_title='Load Queue Occupancy',file_name='plots/occupancy_loadq_'+str(i))
    average_lq_occupancy = np.average(average_lq_occupancy, weights=cycles)

    # Average ready-to-execute strands
    try:
        average_ready_strands = []
        for i in range(num_pe):
            occupancy_time, occupancy_val = list(zip(*logged['PE'][i]['ready_strands']))[0], list(zip(*logged['PE'][i]['ready_strands']))[1]
            # We will truncate the index based on the end of simulation (avoid any later points.)
            idx = truncate_idx(occupancy_time, logged['PE'][i]['end_sim'])
            average_ready_strands.append(np.mean(occupancy_val[:idx]))
            if(i==0 and show):
                plot_graph([occupancy_time[:idx],], [occupancy_val[:idx],], labels=('Cycles', "Ready Strands"), plot_title='Strands Ready/PE',file_name='plots/occupancy_strand_ready_'+str(i))
        average_ready_strands = np.average(average_ready_strands, weights=cycles)
    except:
        average_ready_strands = 'N/A'

    # Instruction Type
    # try:
    # total_instructions = sum(list(logged['PE'][0]['instr_type'].values()))
    # control_inst = logged['PE'][0]['instr_type']['control']
    # arith_inst = logged['PE'][0]['instr_type']['arith']
    # load_inst = logged['PE'][0]['instr_type']['load']
    # print("Control Instructions: {0}, Arith: {1}, load: {2}".format(control_inst/total_instructions, arith_inst/total_instructions, load_inst/total_instructions))
    # except:
        # pass


    # L1-DCache
    hit_rate = [float(logged['L1_dCache'][i]['read_hits'])/(logged['L1_dCache'][i]['read_hits'] + logged['L1_dCache'][i]['read_misses']) for i in range(num_pe)]
    average_hit_rate = 100*np.average(hit_rate, weights=cycles)

    # BTB
    btb_hit_rate = [float(logged['PE'][i]['btb']['hit'])/(logged['PE'][i]['btb']['read']) for i in range(num_pe)]
    average_btb_hitrate = 100*np.average(btb_hit_rate, weights=cycles)
       
    # L1-D MSHR Occupancy
    average_l1d_occupancy = []
    for i in range(num_pe):
        occupancy_time, occupancy_val = list(zip(*logged['L1_dCache'][i]['queue_occupancy']))[0], list(zip(*logged['L1_dCache'][i]['queue_occupancy']))[1]
        # Need to truncaet based on time
        idx = truncate_idx(occupancy_time,logged['PE'][i]['end_sim'])
        if(i==0 and show):
            plot_graph([occupancy_time[:idx],], [ percentagify(occupancy_val[:idx]),], labels=('Cycle', "Occupancy (%)"), plot_title='L1-D MSHR occupancy',file_name='plots/occupancy_l1d_mshr')
        average_l1d_occupancy.append(np.mean(occupancy_val[:idx]))
    average_l1d_occupancy = 100*np.average(average_l1d_occupancy, weights=cycles)

    # L2 Cache
    l2_hit_rate = [float(logged['L2_Cache'][i]['read_hits'])/(logged['L2_Cache'][i]['read_hits'] + logged['L2_Cache'][i]['read_misses']) for i in range(num_pe)]
    average_l2_hit_rate = 100*np.average(l2_hit_rate, weights=cycles)
    # L2 MSHR Occupancy
    average_l2_occupancy = []
    for i in range(num_pe):
        occupancy_time, occupancy_val = list(zip(*logged['L2_Cache'][i]['queue_occupancy']))[0], list(zip(*logged['L2_Cache'][i]['queue_occupancy']))[1]
        # Need to truncaet based on time
        idx = truncate_idx(occupancy_time,logged['PE'][i]['end_sim'])
        if(i==0 and show):
            plot_graph([occupancy_time[:idx],], [ percentagify(occupancy_val[:idx]),], labels=('Cycle', "Occupancy (%)"), plot_title='L2 MSHR occupancy',file_name='plots/occupancy_l2_mshr')
        average_l2_occupancy.append(np.mean(occupancy_val[:idx]))
    average_l2_occupancy = 100*np.average(average_l2_occupancy, weights=cycles)
    
    # Average all the metrics across PEs
    average_compute_util = 100*np.average(compute_util, weights=cycles)
    average_mem_wait = 100*np.average(mem_wait, weights=cycles)
    average_latency = np.average(access_latency, weights=cycles)
    total_cycles = max(cycles)


    # L3 Cache
    l3_hitrate = 100*float(logged['L3_Cache']['read_hits'])/(logged['L3_Cache']['read_hits'] + logged['L3_Cache']['read_misses'])

    # L3-MSHR
    occupancy_time, occupancy_val = list(zip(*logged['L3_Cache']['queue_occupancy']))[0], list(zip(*logged['L3_Cache']['queue_occupancy']))[1]
    # Need to truncaet based on time
    idx = truncate_idx(occupancy_time,logged['end_sim'])
    if(show):
        plot_graph([occupancy_time[:idx],], [ percentagify(occupancy_val[:idx]),], labels=('Cycle', "Occupancy (%)"), plot_title='L3 MSHR occupancy',file_name='plots/occupancy_l3_mshr')
    average_l3_occupancy = 100*np.mean(occupancy_val[:idx])

    ########## On Chip Network ##########

    # Utilization

    occupancy_time, occupancy_val = list(zip(*logged['L2_L3_NoC']['utilization']))[0], list(zip(*logged['L2_L3_NoC']['utilization']))[1]
    # Need to truncaet based on time
    post_idx = truncate_idx(occupancy_time,logged['end_sim'])
    pre_idx = truncate_idx(occupancy_time,logged['start_sim'])
    if(i==0 and show):
        plot_graph([occupancy_time[pre_idx:post_idx],], [ percentagify(occupancy_val[pre_idx:post_idx]),],  labels=('Cycle', "Bandwidth Usage (%)"), plot_title='On-chip Bandwidth usage',file_name='plots/occupancy_onchip_bw')
    average_onchip_noc_usage = 100*np.mean(occupancy_val[pre_idx:post_idx])
    
    # Bytes transferred
    total_bytes_onchip = logged['L2_L3_NoC']['read_bytes_transferred'] + logged['L2_L3_NoC']['write_bytes_transferred']
    read_bytes_onchip = 100*float(logged['L2_L3_NoC']['read_bytes_transferred'])/total_bytes_onchip

    ########## L2-DRAM NoC (Off Chip Network) ##########

    # Utilization

    occupancy_time, occupancy_val = list(zip(*logged['L3_DRAM_NoC']['utilization']))[0], list(zip(*logged['L3_DRAM_NoC']['utilization']))[1]
    # Need to truncaet based on time
    post_idx = truncate_idx(occupancy_time,logged['end_sim'])
    pre_idx = truncate_idx(occupancy_time,logged['start_sim'])
    if(i==0 and show):
        plot_graph([occupancy_time[pre_idx:post_idx],], [ percentagify(occupancy_val[pre_idx:post_idx]),],  labels=('Cycle', "Bandwidth Usage (%)"), plot_title='Off-chip Bandwidth usage',file_name='plots/occupancy_offchip_bw')
    average_l3_dram_noc_usage = 100*np.mean(occupancy_val[pre_idx:post_idx])
    
    # Bytes transferred
    total_bytes_l3_dram = logged['L3_DRAM_NoC']['read_bytes_transferred'] + logged['L3_DRAM_NoC']['write_bytes_transferred']
    read_bytes_l3_dram = 100*float(logged['L3_DRAM_NoC']['read_bytes_transferred'])/total_bytes_l3_dram

    if(show):
        print("\n\n\t\t ---- STATS ---- \n\n")
        print("Average Compute Util: {0}%, \nPercentage of time waiting for mem {1}%, \nAverage Mem access Latency {3} cycles, \
                \nRegfile Occupancy: {4}%, \nAverage Outstanding Loads: {12}, \nAverage Active Strands: {5}, \nL1D Hitrate {2}%, \nl1d queue occupancy: {6}%, \nBTB Hitrate: {11}%\
                \nL2 Hitrate {9}%, \nl2 queue occupancy: {10}%, \
                \ntotal cycles {7}, \nStrands ready to execute {8}"\
                .format(average_compute_util, average_mem_wait, average_hit_rate, average_latency, average_regfile_occupancy, average_thread_occupancy, average_l1d_occupancy, total_cycles, \
                        average_ready_strands, average_l2_hit_rate, average_l2_occupancy, average_btb_hitrate, average_lq_occupancy))
        print("L3 Hitrate {0}%, \nL3 Outstanding Queue Occupancy {1}%, \nOn-chip NoC Utilization {2}%, \nOff-chip NoC Utilization {3}%,  \nOn-Chip Total bytes transferred(MBs): {4} ({5} % reads), \nOn-Chip Total bytes transferred(MBs): {6} ({7} % reads) "\
            .format(l3_hitrate, average_l2_occupancy, average_onchip_noc_usage, average_l3_dram_noc_usage, total_bytes_onchip/(1024*1024), read_bytes_onchip, total_bytes_l3_dram/(1024*1024), read_bytes_l3_dram ))

    keys = ["util", "mem_wait", "amat", "regfile_occupancy", "active_strands", "l1_hitrate", "l1_occupancy", 'btb_hitrate', "l2_hitrate", "l2_occupancy", "cycles", 'strands_ready', 'outstanding_loads']
    values = [average_compute_util, average_mem_wait,  average_latency, average_regfile_occupancy, average_thread_occupancy, average_hit_rate, average_l1d_occupancy, \
                        average_btb_hitrate, average_l2_hit_rate, average_l2_occupancy, total_cycles, average_ready_strands, average_lq_occupancy,]
    keys += ['l3_hitrate', 'average_l2_occupancy', 'average_onchip_noc_usage', 'average_l3_dram_noc_usage', 'total_bytes_onchip', 'read_bytes_onchip', 'total_bytes_l3_dram', 'read_bytes_l3_dram']
    values += [l3_hitrate, average_l2_occupancy, average_onchip_noc_usage, average_l3_dram_noc_usage, total_bytes_onchip/(1024*1024), read_bytes_onchip, total_bytes_l3_dram/(1024*1024), read_bytes_l3_dram]

    stats_dict = dict(zip(keys, values))
    return stats_dict

    # return average_compute_util, average_hit_rate, total_cycles, average_l1d_occupancy, average_regfile_occupancy, average_thread_occupancy, l3_hitrate, average_l3_occupancy, average_onchip_noc_usage, average_l3_dram_noc_usage, average_latency


if __name__ == "__main__":
    data = np.load(sys.argv[1], allow_pickle=True).item()
    show = len(sys.argv)>2
    print("Plotting graphs\n\n" if(show) else "Not Plotting Graphs\n\n")
    postprocess(data, show)
