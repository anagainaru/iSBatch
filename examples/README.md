<img src="https://github.com/anagainaru/iSBatch/blob/master/docs/logo.png" align="right" alt="Logo" width="250"/>

# iSBatch (Batch Scheduler Interface)

This folder contains examples on how to use iSBatch. Details on the implementation can be found in the docs folder.

**1. Simple example for computing the sequence of requests for default parameters**

```bash
Usage: get_sequence.py log_file

> python get_sequence.py logs/truncnorm.in 
Request sequence: [(11.215230398339102, 0), (13.33928544239686, 0)]
```

By default the HPC cost model is being used with the NeverCheckpoint strategy (for applications that do not want to use checkpointing)

**2. Example on how to interpret how good a sequence is**

```bash
Usage: check_sequence_goodness.py log_file training_size

> python check_sequence_goodness.py logs/truncnorm.in 10
Request sequence based on the entire dataset: [(10.733036066210392, 0), (13.33928544239686, 0)] Cost 11.75
Request sequence based on training: [(13.33928544239686, 0)]
Sequence cost: 13.34 (within 13.56% of optimal)
```

Using the default HPC cost model and the NeverCheckpoint strategy for checkpointing, the example computes the cost of a sequence
on the same data that was used to compute the sequence (the entire history log) and compares it to the cost of the sequence
computed using the first `training_size` elements of the history. 

**3. Example on how to use the dynamic checkpoint**

```bash
Usage: use_dynamic_checkpointing.py log_file

> python use_dynamic_checkpointing.py logs/truncnorm.in 
Request sequence (static checkpoint): [(9.729941416301234, 1), (2.0187814273535434, 1), (1.454875089956344, 1), (0.13568750878573965, 0)]
Request sequence (dynamic checkpoint): [(12.217778935513127, 1), (0.9858189980979937, 1), (0.13568750878573965, 0)]
```

Using the HPC cost model and the AlwaysCheckpoint strategy (meaning all the submissions except the last will include a checkpoint)
the example compares the memory static checkpoint model (where the checkpoint/restart cost have the same value regardless when
the snapshot is being taken) to a memory dynamic checkpoint model (the snapshot size changes throughout the lifetime of the 
application)

**4. Jupyter notebook for ploting the sequence of requests**

The plots include the intial historic data (histogram) and the corresponding CDF. The figures are show in the root README.md file
of the repository ([Here](https://github.com/anagainaru/iSBatch/blob/master/README.md))
