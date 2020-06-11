import glob
import numpy as np
import sys

def group_memory_footprint(memory_list, th_size):
    # group similar consecutive entries in the memory list (ts, size)
    ts = 0
    size_list = [memory_list[0][1]]
    group_footprint = []
    for i in range(1, len(memory_list)):
        if abs(memory_list[i][1] - memory_list[i-1][1]) > th_size:
            if len(group_footprint) == 0 or \
                abs(np.mean(size_list) - group_footprint[-1][1]) > th_size:
                group_footprint.append([ts, np.mean(size_list)])
                ts = memory_list[i][0]
                size_list = []
        size_list.append(memory_list[i][1])
    return group_footprint

def pad_array(a, n):
    if len(a) >= n:
        return a
    pad_a = np.zeros((n, 2))
    if len(a) > 0:
        pad_a[:len(a), :] = a
    return pad_a

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s memory_file_prefix [output_file]" %(sys.argv[0]))
        exit()

    memory_log = []
    for file_name in glob.glob(sys.argv[1]+"*"):
        print(file_name)
        data = np.loadtxt(file_name, delimiter=',')
        maxlen = max(len(data), len(memory_log))
        memory_log = pad_array(memory_log, maxlen)
        data = pad_array(data, maxlen)
        memory_log = np.maximum(data, memory_log)
    
    # threshold of 200 MB
    memory_log = group_memory_footprint(memory_log, 204800)
    
    # store data in MB format
    memory_log = np.array([[i[0], i[1]/1024] for i in memory_log])
    outfile = "out_memory.log"
    if len(sys.argv) == 3:
        outfile = sys.argv[2]
    np.savetxt(outfile, memory_log, delimiter=',', fmt='%.2f')
