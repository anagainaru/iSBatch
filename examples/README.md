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

Examples for generating an aggregated memory file used by the dynamic checkpoint can be found in the `logs` folder: `SLANT_run[id].mem`. The script that will parse all these logs and will generate one unique file with the maximum footprint at every given moment based on the logs can be used in the following manner:

```bash
Usage: prepare_memory_log.py memory_file_prefix [output_file]

> python prepare_memory_log.py SLANT_run SLANT_memory.log 
SLANT_run1.mem
SLANT_run2.mem
SLANT_run3.mem
SLANT_run4.mem

The output is generated in SLANT_memory.log
```
This file can later be used together with a walltime log for the same application to generate the sequences of requests:

```bash
Usage: use_dynamic_checkpointing.py walltime_log_file memory_file

> python use_dynamic_checkpointing.py logs/SLANT_walltime.log logs/SLANT_memory.log 
Request sequence (static checkpoint): [(9185.0, 1), (405.0, 0)]
Request sequence (dynamic checkpoint): [(8635.0, 1), (550.0, 1), (405.0, 0)]
```

Using the HPC cost model and the AlwaysCheckpoint strategy (meaning all the submissions except the last will include a checkpoint) the example compares the memory static checkpoint model (where the checkpoint/restart cost have the same value regardless when the snapshot is being taken, specifically the max memory footprint) to a memory dynamic checkpoint model (the snapshot size changes throughout the lifetime of the 
application)

**4. Jupyter notebook for ploting the sequence of requests**

The plots include the intial historic data (histogram) and the corresponding CDF. The figures are show in the root README.md file
of the repository ([Here](https://github.com/anagainaru/iSBatch/blob/master/README.md))
