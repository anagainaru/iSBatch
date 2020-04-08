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
    training = data[:train_size] + [max(data)]

    # compute the requests based on the training data
    wl = rqs.ResourceEstimator(training)
    sequence = wl.compute_request_sequence()
    cost = wl.compute_sequence_cost(sequence, data)

    print("Request sequence based on training: %s" %(sequence))
    print("Sequence cost: %s" %(cost))
