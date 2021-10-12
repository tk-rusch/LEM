<h1 align='center'> Long Expressive Memory for Sequence Modeling </h1>

This repository contains the implementation to reproduce the numerical experiments 
of the paper [Long Expressive Memory for Sequence Modeling](https://arxiv.org/abs/2110.04744)



## Requirements
To setup the environment, install the requirements using *python 3.7.10* with
```
pip install -r requirements.txt
```

## Experiments

This repository contains the codes to reproduce the results 
of the following experiments for the proposed LEM:

  - **Adding Task** 
  - **EigenWorms** 
  - **FitzHugh-Nagumo** 
  - **Google12 (V2)**
  - **Heart rate prediction**
  - **noisy CIFAR-10**
  - **Permuted Sequential MNIST**
  - **PennTree Bank (word- and char-level)**
  - **Sequential MNIST**

All datasets get either synthetically generated or downloaded automatically, 
providing a full data pipline for each experiment. <br>
Note that the PTB experiments are based on the open-source tensorflow project 
[LAnguage Modelling Benchmarks (lamb)](https://github.com/deepmind/lamb)
for tuning and testing Tensorflow language models. 
In order to reproduce these experiments, 
please follow the steps explained in the *README* of the PTB directory.

## Results
The results of LEM for each of the experiments are:
<table>
  <tr>
    <td> Experiment </td>
    <td> Result </td>
  </tr>
  <tr>
    <td>EigenWorms </td>
    <td> 92.3% mean test accuracy</td>
  </tr>
  <tr>
    <td>FitzHugh-Nagumo </td>
    <td>  0.002 test L^2 loss </td>
  </tr>
    <tr>
    <td>Google12 (V2)</td>
    <td> 95.7% test accuracy </td>
  </tr>
  <tr>
    <td>Heart rate prediction</td>
    <td> 0.85 test L^2 loss </td>
  </tr>
  <tr>
    <td>noisy CIFAR-10</td>
    <td> 60.5% test accuracy  </td>
  </tr>
  <tr>
    <td>Permuted Sequential MNIST</td>
    <td> 96.6% test accuracy </td>
  </tr>
<tr>
    <td>PennTree Bank char-level (single layer)</td>
    <td> 1.25 test bits-per-character </td>
  </tr>
<tr>
    <td>PennTree Bank word-level (single layer)</td>
    <td> 72.8 test perplexity </td>
  </tr>
<tr>
    <td>Sequential MNIST</td>
    <td> 99.5% test accuracy </td>
  </tr>
</table>
