import sys
sys.path.append("..")
import iSBatch as rqs
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: %s log_file training_size" %(sys.argv[0]))
        exit()

    file_name = sys.argv[1]
    train_size = int(sys.argv[2])

    data = np.loadtxt(file_name, delimiter=' ')
    assert (len(data) > train_size), "Training size exceeds the total dataset"
    training = list(data[:train_size]) + [max(data)]

    # compute the requests based on the entire data
    wl = rqs.ResourceEstimator(data)
    cl = rqs.ClusterCosts(1, 0, 0)
    sequence = wl.compute_request_sequence(cluster_cost=cl)
    # cost value will be used as reference as optimal
    cost_opt = wl.compute_sequence_cost(sequence, data, cluster_cost=cl)

    print("Request sequence based on the entire dataset: %s Cost %.2f" % (
        sequence, cost_opt))

    # compute the requests based on the training data
    wl = rqs.ResourceEstimator(training)
    sequence = wl.compute_request_sequence(cluster_cost=cl)
    cost = wl.compute_sequence_cost(sequence, data, cluster_cost=cl)

    print("Request sequence based on training: %s\n"\
          "Sequence cost: %.2f (within %.2f%% of optimal)" % (
              sequence, cost, (cost-cost_opt)*100/cost_opt))
