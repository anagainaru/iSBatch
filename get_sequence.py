import HPCRequest as rqs
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s log_file" %(sys.argv[0]))
        exit()

    file_name = sys.argv[1]
    history = np.loadtxt(file_name, delimiter=' ')
    wl = rqs.Workload(history)
    sequence = wl.compute_request_sequence(alpha=1, beta=1, gamma=0)
    print("Request sequence: %s" %(sequence))