# HPC Walltime Request

Code for computing the optimal request time for a stochastic application when submitted on an HPC scheduler, based on historic information about previous runs.
Any code wanting to use this code has to include:
- `import HPCRequest`

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
*Example discrete vs interpolation CDF and sequences (for 10 and 300 datasets)*


### 3. [Optional] Compute the cost of a sequence of requests for new data

Compute the cost of a given sequence by creating a Cost object based on the sequence and runing it on the new data. The cost represents the average time used by each submission for all reservations. This time represents all the failed reservation together with the sucessful one. For example, for two submissions one of 10 and another of 15 hours, the cost of the sequence [8, 11, 16] is the average between `8 + 10` (the first submission will fail when requesting 8hs and will succeed the second time) and `8 + 11 + 15`.

```python
cost = wf.compute_sequence_cost(sequence, new_data)
```
