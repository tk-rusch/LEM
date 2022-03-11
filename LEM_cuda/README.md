<h1 align='center'>  Fast C++/CUDA PyTorch extension of LEM </h1>

We provide a fast (**3-5 times faster than standard PyTorch+cuda on GPUs**) 
CUDA extension of LEM. 
It only needs to be compiled once and can then be used wherever and whenever wanted.

# Compiling the extensions
Navigate into `LEM_cuda_extension` and simply run 

```
python setup.py install
```

If the compilation was successful, you should be able to find it in your installed python packages,
e.g. with pip:

```
pip list
```
should show a package called `lem-cuda` with version `0.0.0`. 

### Trouble shooting
Please make sure you have the correct pytorch cuda version 
installed for your machine (e.g. cuda 10.1 or cuda 10.2 and so on).

On linux fedora (and probably other linux distributions)
you might get a warning that the extension was 
compiled with a different C++ compiler than pytorch was build with. If you have the `gcc-c++` compiler installed, 
you can ignore these warnings (might be some name miss-match inside pytorch).

If it still doesn't work, please feel free to open an issue.

# Usage
Once compiled, it can be used in standard PyTorch by defining the compiled extensions as:

```python
import math
import torch
from torch import nn
import lem_cuda

class LEMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, weights_lin_z, bias,
                bias_lin_z, initial_y_state, initial_z_state, dt):
        all_y_states, all_z_states, all_X, all_X2, \
        all_multi_scales, all_lin_new_z_state = lem_cuda.forward(inputs, weights,
                                                                 weights_lin_z, bias, bias_lin_z,
                                                                 initial_y_state, initial_z_state, dt)
        ctx.save_for_backward(all_X, all_X2, all_multi_scales,
                              all_lin_new_z_state, weights, weights_lin_z, bias,
                              bias_lin_z, initial_y_state, initial_z_state, dt)
        return all_y_states, all_z_states

    @staticmethod
    def backward(ctx, grad_y_states, grad_z_states):
        outputs = lem_cuda.backward(grad_y_states.contiguous(), grad_z_states.contiguous(), *ctx.saved_tensors)
        d_inputs, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state = outputs
        return None, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state, None


class LEMcuda(torch.nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMcuda, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.weights = torch.nn.Parameter(torch.empty(3 * nhid, ninp + nhid))
        self.weights_lin_z = torch.nn.Parameter(torch.empty(nhid, ninp + nhid))
        self.bias = torch.nn.Parameter(torch.empty(3 * nhid))
        self.bias_lin_z = torch.nn.Parameter(torch.empty(nhid))
        self.dt = torch.tensor(dt).reshape(1, 1).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, states=None):
        if states is None:
            y = input.data.new(input.size(1), self.nhid).zero_()
            z = input.data.new(input.size(1), self.nhid).zero_()
            states = (y,z)
        return LEMFunction.apply(input.contiguous(), self.weights, 
                                 self.weights_lin_z, self.bias, 
                                 self.bias_lin_z, *states, self.dt)
```

`LEMcuda` can now be used like `nn.LSTM` or `nn.GRU`, e.g.

```python
class LEM(torch.nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.):
        super(LEM, self).__init__()
        self.rnn = LEMcuda(ninp,nhid,dt)
        self.classifier = nn.Linear(nhid,nout)

    def forward(self, input):
        all_y, all_z = self.rnn(input)
        out = self.classifier(all_y[-1])
        return out
```

Note that the input to `LEMcuda` has to be in the following shape: `time x batch x input_dim`, 
i.e. we don't allow for batch_first.

# Example
Finally, we provide a simple example on how to use this extension in practice.
To do so, we consider learning the FitzHughNagumo model 
(same example as in our paper and `src`, but this time using the fast compiled LEM extensions).
After having successfully compiled the extensions, you can simply run 

```
python example/FitzHughNagumo_task.py
```
