# noisy CIFAR-10
## Usage
The dataset is downloaded automatically through torchvision.

To start the training with LEM, simply run:
```
python nCIFAR_task.py [args]
```

Options:
- nhid : hidden size
- epochs : max number of epochs
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- dt : parameter $Delta t$ of LEM
- seed : random seed

The log of our best run can 
be found in the result directory.
