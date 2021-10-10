# PennTree Bank (word-level and char-level)
## Preparation
This implementation is based on the open-source tensorflow project 
[LAnguage Modelling Benchmarks (lamb)](https://github.com/deepmind/lamb)
for tuning and testing Tensorflow language models.

To reproduce our LEM results for PTB word-level and char-level, 
simply download the lamb package and exchange the file *tiled_lstm.py* 
with our own version of it before installation.
There, instead of an LSTM cell, a **LEM cell** is implemented.

We provide the config files to reprocude both results, i.e. 
PTB word-level as well as PTB char-level, in the directory **LEM_configs**.

In order to start training, please copy the **LEM_configs** directory 
to **lamb/experiment/mogrifier/** of the lamb package.
## Usage
### Training
After preparing and installing the code, you can start training 
by first navigating to the directory **lamb/experiment/mogrifier/** 
and then simply running the following commands,<br />
**Word-level:**
```
bash train_ptb.sh run <train_directory> LEM_configs/LEM_config_word
```
**Character-level:**
```
bash train_ptb_char.sh run <train_directory> LEM_configs/LEM_config_char
```

### Testing
First, change the *test.sh* in **lamb/experiment/** with 
our slightly changed version of it (bug fix + unset MC evaluating).
After that and after the training has finished, you can 
test the trained LEM by navigating again to **lamb/experiment/mogrifier/** 
and run the following commands (same for word and char-level):
```
bash ../test.sh run <test_directory> ./<train_directory>_<random_signature>/
```
