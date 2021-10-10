# FitzHugh-Nagumo
## Data preparation
We generate the training, testing and validation data by solving 
the FitzHugh-Nagumo ODE (parameters specified in the paper) with the explicit Runge-Kutta method of order5(4) 
(default solver of scipy's integrate.solve_ivp method) for different random initial values in the *data.py* file.

## Usage
To start the training with LEM, simply run:
```
python FitzHughNagumo_task.py [args]
```

Options:
- nhid : hidden size
- epochs : max number of epochs
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- seed : random seed

The log of our best run can 
be found in the result directory.
