import sys
sys.path.append("..")
import iSBatch as rqs
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s log_file" %(sys.argv[0]))
        exit()

    file_name = sys.argv[1]
    history = np.loadtxt(file_name, delimiter=' ')

    # set the cluster cost model
    check_model = rqs.DynamicCheckpointMemoryModel([(10, 0), (1, 12)])
    cl_cost = rqs.ClusterCosts(checkpoint_memory_model=check_model)
    wl = rqs.ResourceEstimator(history,
                               CR_strategy=rqs.CRStrategy.AlwaysCheckpoint)
    sequence = wl.compute_request_sequence()
    print("Request sequence (static checkpoint): %s" %(sequence))
    
    sequence = wl.compute_request_sequence(cluster_cost=cl_cost)
    print("Request sequence (dynamic checkpoint): %s" %(sequence))
