import sys
sys.path.append("..")
import iSBatch as rqs
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: %s walltime_log_file memory_file" %(sys.argv[0]))
        exit()

    file_name = sys.argv[1]
    history = np.loadtxt(file_name, delimiter=' ')
    memory_footprint = np.loadtxt(sys.argv[2], delimiter=',')

    # set the cluster cost model
    params = rqs.ResourceParameters()
    params.CR_strategy = rqs.CRStrategy.AlwaysCheckpoint
    check_model = rqs.StaticCheckpointMemoryModel(
            checkpoint_cost=np.max(memory_footprint),
            restart_cost=np.max(memory_footprint))
    cl_cost = rqs.ClusterCosts(checkpoint_memory_model=check_model)
    wl = rqs.ResourceEstimator(history, params=params)
    sequence = wl.compute_request_sequence(cluster_cost=cl_cost)
    print("Request sequence (static checkpoint): %s" %(sequence))
    
    check_model = rqs.DynamicCheckpointMemoryModel(memory_footprint)
    cl_cost = rqs.ClusterCosts(checkpoint_memory_model=check_model)
    sequence = wl.compute_request_sequence(cluster_cost=cl_cost)
    print("Request sequence (dynamic checkpoint): %s" %(sequence))
