<img src="./docs/logo.png" align="right" alt="Logo" width="250"/>

# iSBatch (Batch Scheduler Interface)

iSBatch is a python package that generates resource requests for an application to be used when submitting it on an HPC cluster. *Currently node hours is the only supported resource.*

iSBatch requires past execution times for an application (it is recommanded to have at least 10 runs) as well as basic informaion on the HPC cluster and generates a sequence of requests to be used when submitting on the cluster (if the first request is an under-estimation of the real execution time, the second request should be used and so on)

**Table of contents**
- [Brief theory behind the generated requests](#theory)
- [Usage](#usage)
- [Papers describing internal algorithms](#papers)
- [Code documentation](./docs/README.md)

## Theory

iSBach is computing the optimal request time for a stochastic application based on historic information about previous runs. The following theorem is used to compute the sequence of requests: 

![Optimal sequence](https://github.com/anagainaru/HPCWalltime/blob/master/docs/progdyn.png)

Different alpha, beta, gamma values can be chosen to model large-scale systems. We use:
 - **alpha=1, beta=0, gamma=0** to model the Cloud cost model (*pay for what you reserve*)
 - **alpha=1, beta>=1, gamma=0** for the HPC cost model (*pay for what you reserve* - wait time in the queue - *plus for what you use* - node hours).

The HPC model is chosen by detault.

## Usage

To use this code for generating walltime requests, include:
- `import HPCRequest`

Simple examples on how to use the modules in the library are shown in the jupyter lab file and in the `generate_sequence.py` file. Running this file on an input log (like the ones provided as examples in the `log_examples` folder) will return the execution time you need to use for submission on an HPC system described by the default parameters.

```bash
user:~/iSBatch$ python get_sequence.py log_examples/neuroscience.in 
Request sequence: [(80802.0,), (99019.0,)]

```
To create your own scripts use the following steps:

### 1. Prepare the data

Create a list with historic resource usage information. If walltime is the resource under study, `data` is a list of walltimes for past runs. Create a Workload object with the historic data (and optionally the interpolation method required)

```python
wf = HPCRequest.Workload(data)

wf_inter = HPCRequest.Workload(data, interpolation_model=HPCRequest.DistInterpolation)
```

If you wish to print the CDF of this data, the discrete data (e.g. unique walltimes) and the associated CDF values for each data can be extracted using the compute_cdf function:

```python3
optimal_data, optimal_cdf = wf.compute_cdf()
```

![Example CDF](./docs/discrete_cdf.png)
*Example discrete CDF and data (without using interpolation) - vertical blue lines represent the recommended request times*

### 2. Compute the sequence of requests

The `compute_sequence` function returns the recommended sequence of requests given a historic data. Optionally, the function takes the upper limit expected for the execution time of the application (if nothing is provided, the max(data) will be used as upper limit)

```python
sequence = wf.compute_sequence(max_value=100)
```
For large historic datasets, computing the distribution using the discrete data will give good results. Otherwise, interpolation is needed. 

![Example sequence](./docs/sequence.png)
*Example discrete vs interpolation CDF and sequences*


### 3. [Optional] Compute the cost of a sequence of requests for new data

Compute the cost of a given sequence by creating a Cost object based on the sequence and runing it on the new data. The cost represents the average time used by each submission for all reservations. This time represents all the failed reservation together with the sucessful one. For example, for two submissions one of 10 and another of 15 hours, the cost of the sequence [8, 11, 16] is the average between `8 + 10` (the first submission will fail when requesting 8hs and will succeed the second time) and `8 + 11 + 15`.

```python
cost = wf.compute_sequence_cost(sequence, new_data)
```

## Papers


If you use the resources available here in your work, please cite our paper:

```
@INPROCEEDINGS{8948696,
author={A. {Gainaru} and G. {Pallez}},
booktitle={2019 IEEE/ACM 10th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (ScalA)},
title={Making Speculative Scheduling Robust to Incomplete Data},
year={2019},
volume={},
number={},
pages={62-71},
keywords={HPC scheduling;stochastic applications;perfor- mance modeling;discrete and continuous estimators},
doi={10.1109/ScalA49573.2019.00013},
ISSN={null},
month={Nov},}
```

For details about how to compute the optimal sequence of requests, please consult our paper: <br/>
**Reservation and Checkpointing Strategies for Stochastic Jobs** <br/>
Ana Gainaru, Brice Goglin, Valentin Honoré, Guillaume Pallez, Padma
Raghavan, Yves Robert, Hongyang Sun.
[IPDPS 2020] (Paper: [INRIA technical report](https://hal.inria.fr/hal-02328013/document))

For details about why interpolation is needed when the historic information is low read our paper: <br/>
**Making Speculative Scheduling Robust to Incomplete Data**<br/>
Ana Gainaru, Guillaume Pallez. 
[SCALA@SC 2019] (Paper: [INRIA technical report](https://hal.inria.fr/hal-02158598/document))<br/>

For details on how to adapt the sequence of requests when backfilling is being used: <br/>
**Speculative Scheduling Techniques for Stochastic HPC Applications**<br/>
Ana Gainaru, Guillaume Pallez, Hongyang Sun, Padma Raghavan [ICPP 2019] (Paper: [INRIA technical report](https://hal.inria.fr/hal-02158598/document))


