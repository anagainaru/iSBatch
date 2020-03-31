<img src="./logo.png" align="right" alt="Logo" width="250"/>

# Documentation

![API](https://raw.githubusercontent.com/anagainaru/iSBatch/master/docs/api.png)

The main class of the project is the **ResourceEstimator**. It takes information about the application:
 - A list of past execution runs
 - The checkpointing strategy (one of the options defined by the Checkpoint/Restart Strategy class)
 
 ```python
    def __init__(self, past_runs, interpolation_model=None,
                 CR_strategy=CRStrategy.NeverCheckpoint, verbose=False):
```
 
 Once an object has been instantiate for a given application, the CR strategy or the default interpoltion methods can be changed.
 
