# Adding task
## Usage
As this task is not based on a pre-defined data set, 
the training and test data is synthetically generated.

```
python adding_task.py [args]
```

Options:
- seq_length : length of sequences
- nhid : hidden size
- max_steps : max number of learning steps
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- dt : parameter $Delta t$ of LEM

Logs of the runs for all three sequence lengths (i.e. 2000, 5000, 10000) can 
be found in the results directory.