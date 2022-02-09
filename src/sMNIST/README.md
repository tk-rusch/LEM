# Sequential MNIST
## Usage
The dataset is downloaded automatically through torchvision.

To start the training with LEM, simply run:
```
python sMNIST_task.py [args]
```

Options:
- nhid : hidden size
- epochs : max number of epochs
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- dt : parameter $Delta t$ of LEM
- grad_norm : max norm for gradient clipping
- seed : random seed

The log of our best run can 
be found in the result directory.
