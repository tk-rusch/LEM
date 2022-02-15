<h1 align='center'> Long Expressive Memory for Sequence Modeling<br>
    *ICLR 2022 Spotlight* </h1>


This repository contains the implementation to reproduce the numerical experiments 
of the *ICLR 2022* **[spotlight]** 
paper [Long Expressive Memory for Sequence Modeling](https://openreview.net/forum?id=vwj6aUeocyf)


# Update</h1> 
A fast mixed C++/CUDA PyTorch extension implementation of LEM is 
now available (**3-5 times faster** than LEM using standard PyTorch+cuda on GPUs).

It's as easy to use as PyTorch's `nn.LSTM` or `nn.GRU`.

Follow the steps in `LEM_cuda` directory to compile the extensions. 
We also provide a simple example on how to use it.

If you are planning on using LEM for your own project, I strongly recommend to use the 
compiled PyTorch extension (in `LEM_cuda`)
instead of the standard PyTorch code we provide in `src`. 

# Running the experiments

## Requirements
You can install the requirements using *python 3.7* with
```
conda create --name lem python=3.7
conda activate lem

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



# Citation
If you found our work useful in your research, please cite our paper at:
```bibtex
@inproceedings{rusch2022lem,
  title={Long Expressive Memory for Sequence Modeling},
  author={Rusch, T Konstantin and Mishra, Siddhartha and Erichson, N Benjamin and Mahoney, Michael W},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
(Also consider starring the project on GitHub.)
