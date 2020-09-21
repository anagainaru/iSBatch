import sys
sys.path.append("..")
import iSBatch as rqs
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: %s log_file max_submissions [average_strategy" \
              "(threshold if no argument)]" %(sys.argv[0]))
        exit()

    file_name = sys.argv[1]
    limit = float(sys.argv[2])
    strategy = rqs.LimitStrategy.ThresholdBased
    if len(sys.argv) > 3:
        strategy = rqs.LimitStrategy.AverageBased

    history = np.loadtxt(file_name, delimiter=' ')
    params = rqs.ResourceParameters()
    params.submissions_limit = limit
    params.submissions_limit_strategy = strategy 
    params.CR_strategy = rqs.CRStrategy.NeverCheckpoint
    wl = rqs.ResourceEstimator(history, params=params)
    sequence = wl.compute_request_sequence()
    cost = wl.compute_sequence_cost(sequence, history)
    print("Request sequence: %s" %(sequence))
    print("Sequence cost: %s" %(cost))
    fail = 0
    for i in history:
        compute = 0
        for s in sequence:
            if i > s[0] + compute:
                fail += 1
            if s[1] == 1:
                compute += s[0]
        fail += 1
    print("Average failures", fail / len(history))
